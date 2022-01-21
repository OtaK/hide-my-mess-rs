mod download_rvm;

pub use download_rvm::RvmModelKind;

use crate::error::HideResult;
use tensorflow as tf;

#[derive(Debug)]
struct RvmArgs {
    // Inputs
    src_in: tf::Operation,
    dsr_in: tf::Operation,
    ir1_in: tf::Operation,
    ir2_in: tf::Operation,
    ir3_in: tf::Operation,
    ir4_in: tf::Operation,
    // Outputs
    fgr_out: tf::Operation,
    pha_out: tf::Operation,
    ir1_out: tf::Operation,
    ir2_out: tf::Operation,
    ir3_out: tf::Operation,
    ir4_out: tf::Operation,
}

impl RvmArgs {
    pub fn new(graph: &tf::Graph, signature: &tf::SignatureDef) -> HideResult<Self> {
        // Inputs
        let src_in = Self::get_input_operation_for_param(graph, signature, "src")?;
        let dsr_in = Self::get_input_operation_for_param(graph, signature, "downsample_ratio")?;
        let ir1_in = Self::get_input_operation_for_param(graph, signature, "r1i")?;
        let ir2_in = Self::get_input_operation_for_param(graph, signature, "r2i")?;
        let ir3_in = Self::get_input_operation_for_param(graph, signature, "r3i")?;
        let ir4_in = Self::get_input_operation_for_param(graph, signature, "r4i")?;
        // Outputs
        let fgr_out = Self::get_output_operation_for_param(graph, signature, "fgr")?;
        let pha_out = Self::get_output_operation_for_param(graph, signature, "pha")?;
        let ir1_out = Self::get_output_operation_for_param(graph, signature, "r1o")?;
        let ir2_out = Self::get_output_operation_for_param(graph, signature, "r2o")?;
        let ir3_out = Self::get_output_operation_for_param(graph, signature, "r3o")?;
        let ir4_out = Self::get_output_operation_for_param(graph, signature, "r4o")?;

        Ok(Self {
            src_in,
            dsr_in,
            ir1_in,
            ir2_in,
            ir3_in,
            ir4_in,
            fgr_out,
            pha_out,
            ir1_out,
            ir2_out,
            ir3_out,
            ir4_out,
        })
    }

    #[inline(always)]
    fn get_input_operation_for_param(
        graph: &tf::Graph,
        signature: &tf::SignatureDef,
        param: &str,
    ) -> HideResult<tf::Operation> {
        Ok(graph.operation_by_name_required(&signature.get_input(param)?.name().name)?)
    }

    #[inline(always)]
    fn get_output_operation_for_param(
        graph: &tf::Graph,
        signature: &tf::SignatureDef,
        param: &str,
    ) -> HideResult<tf::Operation> {
        Ok(graph.operation_by_name_required(&signature.get_output(param)?.name().name)?)
    }
}

#[derive(Debug)]
pub struct InitialRecurrentState {
    r1: tf::Tensor<f32>,
    r2: tf::Tensor<f32>,
    r3: tf::Tensor<f32>,
    r4: tf::Tensor<f32>,
}

impl InitialRecurrentState {
    pub fn try_new() -> HideResult<Self> {
        let t = tf::Tensor::new(&[1]).with_values(&[0.])?;
        Ok(Self {
            r1: t.clone(),
            r2: t.clone(),
            r3: t.clone(),
            r4: t,
        })
    }
}

const SUPPORTED_CHANNELS: usize = 3;

#[derive(Debug)]
pub struct RobustVideoMatting {
    #[allow(dead_code)]
    graph: tf::Graph,
    bundle: tf::SavedModelBundle,
    state: InitialRecurrentState,
    fb: Vec<f32>,
    args: RvmArgs,
}

impl RobustVideoMatting {
    pub fn try_init(model_kind: RvmModelKind) -> HideResult<Self> {
        let path = download_rvm::download_rvm_model(model_kind)?;
        let mut graph = tf::Graph::new();

        let bundle =
            tf::SavedModelBundle::load(&tf::SessionOptions::new(), &["serve"], &mut graph, path)?;

        let state = InitialRecurrentState::try_new()?;

        let signature = bundle
            .meta_graph_def()
            .get_signature(tf::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;

        let args = RvmArgs::new(&graph, signature)?;

        Ok(Self {
            graph,
            bundle,
            state,
            fb: vec![],
            args,
        })
    }

    #[inline(always)]
    fn auto_downsample_ratio(height: u32, width: u32) -> f32 {
        let higher_res = std::cmp::max(height, width);
        if higher_res >= 512 {
            512. / (higher_res as f32 * 1.06)
        } else {
            1.
        }
    }

    fn normalize_frame(&mut self, frame: &[u8], width: usize, height: usize) {
        let buffer_size = width * height * SUPPORTED_CHANNELS;
        self.fb.reserve_exact(buffer_size);
        if self.fb.len() < self.fb.capacity() {
            self.fb.resize(buffer_size, 0.);
        }
        let frame_normalized = frame.chunks_exact(4).flat_map(|rgba| {
            [
                rgba[0] as f32 / 255.,
                rgba[1] as f32 / 255.,
                rgba[2] as f32 / 255.,
            ]
        });

        for (fbn, np) in self.fb.iter_mut().zip(frame_normalized) {
            *fbn = np;
        }
    }

    pub fn infer(&mut self, frame: &mut [u8], (width, height): (u32, u32)) -> HideResult<()> {
        self.normalize_frame(frame, width as usize, height as usize);

        // Intermediate tensors
        let frame_tensor: tf::Tensor<f32> =
            tf::Tensor::new(&[1, height as u64, width as u64, SUPPORTED_CHANNELS as u64])
                .with_values(&self.fb)?;

        let dsr_tensor = tf::Tensor::from(Self::auto_downsample_ratio(height, width));

        // Input args
        let mut args = tf::SessionRunArgs::new();
        args.add_feed(&self.args.src_in, 0, &frame_tensor);
        args.add_feed(&self.args.ir1_in, 0, &self.state.r1);
        args.add_feed(&self.args.ir2_in, 0, &self.state.r2);
        args.add_feed(&self.args.ir3_in, 0, &self.state.r3);
        args.add_feed(&self.args.ir4_in, 0, &self.state.r4);
        args.add_feed(&self.args.dsr_in, 0, &dsr_tensor);

        // Output tokens
        let fgr_out_token = args.request_fetch(&self.args.fgr_out, 0);
        let pha_out_token = args.request_fetch(&self.args.pha_out, 1);
        let ir1_out_token = args.request_fetch(&self.args.ir1_out, 2);
        let ir2_out_token = args.request_fetch(&self.args.ir2_out, 3);
        let ir3_out_token = args.request_fetch(&self.args.ir3_out, 4);
        let ir4_out_token = args.request_fetch(&self.args.ir4_out, 5);

        // Hehe
        self.bundle.session.run(&mut args)?;

        // Outputs
        let fgr: tf::Tensor<f32> = args.fetch(fgr_out_token)?;
        let pha: tf::Tensor<f32> = args.fetch(pha_out_token)?;
        let ir1: tf::Tensor<f32> = args.fetch(ir1_out_token)?;
        let ir2: tf::Tensor<f32> = args.fetch(ir2_out_token)?;
        let ir3: tf::Tensor<f32> = args.fetch(ir3_out_token)?;
        let ir4: tf::Tensor<f32> = args.fetch(ir4_out_token)?;

        // Not hehe
        drop(args);

        // Update recurrent states
        self.state.r1 = ir1;
        self.state.r2 = ir2;
        self.state.r3 = ir3;
        self.state.r4 = ir4;

        // Fish for outputs
        log::debug!("fgr: {:#?}", fgr);
        log::debug!("pha: {:#?}", pha);
        log::debug!("state: {:#?}", self.state);

        let infer_iter = fgr
            .chunks_exact(SUPPORTED_CHANNELS)
            .into_iter()
            .zip(pha.into_iter())
            .flat_map(|(rgb, a)| {
                [
                    (rgb[0] * 255.) as u8,
                    (rgb[1] * 255.) as u8,
                    (rgb[2] * 255.) as u8,
                    (*a * 255.) as u8,
                ]
            });

        for (frame_px, infer_px) in frame.iter_mut().zip(infer_iter) {
            *frame_px = infer_px;
        }

        Ok(())
    }
}
