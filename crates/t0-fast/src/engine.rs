//! Owns the wgpu device/queue and the ten compute pipelines used by
//! `model.rs`. Every pipeline uses an auto-derived (`layout: None`) bind
//! group layout, group 0, bindings in declaration order matching each
//! `.wgsl` file. No fusion feature, no Burn tensor anywhere in this crate --
//! see `crates/t0-fast/README.md`.

use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub linear: wgpu::ComputePipeline,
    pub add_inplace: wgpu::ComputePipeline,
    pub gather_add: wgpu::ComputePipeline,
    pub rmsnorm_full: wgpu::ComputePipeline,
    pub rmsnorm_qk: wgpu::ComputePipeline,
    pub rope: wgpu::ComputePipeline,
    pub attention: wgpu::ComputePipeline,
    pub transpose_outer: wgpu::ComputePipeline,
    pub silu_mul: wgpu::ComputePipeline,
    pub quantile_head: wgpu::ComputePipeline,
}

fn make_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Engine {
    pub async fn new_async() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("t0-fast"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu device: {e}"))?;

        Ok(Engine {
            linear: make_pipeline(&device, "linear", include_str!("shaders/linear.wgsl")),
            add_inplace: make_pipeline(&device, "add_inplace", include_str!("shaders/add_inplace.wgsl")),
            gather_add: make_pipeline(&device, "gather_add", include_str!("shaders/gather_add.wgsl")),
            rmsnorm_full: make_pipeline(&device, "rmsnorm_full", include_str!("shaders/rmsnorm_full.wgsl")),
            rmsnorm_qk: make_pipeline(&device, "rmsnorm_qk", include_str!("shaders/rmsnorm_qk.wgsl")),
            rope: make_pipeline(&device, "rope", include_str!("shaders/rope.wgsl")),
            attention: make_pipeline(&device, "attention", include_str!("shaders/attention.wgsl")),
            transpose_outer: make_pipeline(&device, "transpose_outer", include_str!("shaders/transpose_outer.wgsl")),
            silu_mul: make_pipeline(&device, "silu_mul", include_str!("shaders/silu_mul.wgsl")),
            quantile_head: make_pipeline(&device, "quantile_head", include_str!("shaders/quantile_head.wgsl")),
            device,
            queue,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub fn buf_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn buf_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn buf_empty(&self, len_f32: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len_f32.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn buf_uniform<T: bytemuck::Pod>(&self, data: T, label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    pub fn bind_group(&self, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries,
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        wgs: (u32, u32, u32),
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);
    }

    /// One copy-to-staging + map + read. The single readback per forward
    /// pass (see `model.rs::forward_async`) -- everything upstream of this
    /// call stays on the GPU in one command encoder.
    pub async fn read_buffer(&self, buf: &wgpu::Buffer, len_f32: usize) -> Vec<f32> {
        let size = (len_f32 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}
