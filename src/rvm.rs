use crate::error::HideResult;
use tensorflow as tf;

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

#[derive(Debug)]
pub struct RobustVideoMatting {
    graph: tf::Graph,
    bundle: tf::SavedModelBundle,
    state: InitialRecurrentState,
}

impl RobustVideoMatting {
    pub fn try_init() -> HideResult<Self> {
        let mut graph = tf::Graph::new();

        let bundle = tf::SavedModelBundle::load(
            &tf::SessionOptions::new(),
            &["serve"],
            &mut graph,
            "./models/rvm_mobilenetv3_tf",
        )?;

        let state = InitialRecurrentState::try_new()?;

        Ok(Self {
            graph,
            bundle,
            state,
        })
    }

    #[inline(always)]
    fn auto_downsample_ratio(height: u32, width: u32) -> f32 {
        let higher_res = std::cmp::max(height, width);
        if higher_res >= 512 {
            512. / higher_res as f32
        } else {
            1.
        }
    }

    #[inline(always)]
    fn get_input_operation_for_param(&self, signature: &tf::SignatureDef, param: &str) -> HideResult<tf::Operation> {
        Ok(self.graph.operation_by_name_required(&signature.get_input(param)?.name().name)?)
    }

    #[inline(always)]
    fn get_output_operation_for_param(&self, signature: &tf::SignatureDef, param: &str) -> HideResult<tf::Operation> {
        Ok(self.graph.operation_by_name_required(&signature.get_output(param)?.name().name)?)
    }

    pub fn run(&mut self, frame: &[f32], (channels, width, height): (u32, u32, u32)) -> HideResult<Vec<f32>> {
        let signature = self.bundle
            .meta_graph_def()
            .get_signature(tf::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;

        log::debug!("Signature: {:#?}", signature);

        // Inputs
        let src_in = self.get_input_operation_for_param(signature, "src")?;
        let dsr_in = self.get_input_operation_for_param(signature, "downsample_ratio")?;
        let ir1_in = self.get_input_operation_for_param(signature, "r1i")?;
        let ir2_in = self.get_input_operation_for_param(signature, "r2i")?;
        let ir3_in = self.get_input_operation_for_param(signature, "r3i")?;
        let ir4_in = self.get_input_operation_for_param(signature, "r4i")?;

        // Outputs
        let fgr_out = self.get_output_operation_for_param(signature, "fgr")?;
        let pha_out = self.get_output_operation_for_param(signature, "pha")?;
        let ir1_out = self.get_output_operation_for_param(signature, "r1o")?;
        let ir2_out = self.get_output_operation_for_param(signature, "r2o")?;
        let ir3_out = self.get_output_operation_for_param(signature, "r3o")?;
        let ir4_out = self.get_output_operation_for_param(signature, "r4o")?;

        // Intermediate tensors
        let frame_tensor: tf::Tensor<f32> = tf::Tensor::new(&[1, height as u64, width as u64, channels as u64])
            .with_values(frame)?;

        let dsr_tensor = tf::Tensor::from(Self::auto_downsample_ratio(height, width));

        // Input args
        let mut args = tf::SessionRunArgs::new();
        args.add_feed(&src_in, 0, &frame_tensor);
        args.add_feed(&ir1_in, 0, &self.state.r1);
        args.add_feed(&ir2_in, 0, &self.state.r2);
        args.add_feed(&ir3_in, 0, &self.state.r3);
        args.add_feed(&ir4_in, 0, &self.state.r4);
        args.add_feed(&dsr_in, 0, &dsr_tensor);

        // Output tokens
        let fgr_out_token = args.request_fetch(&fgr_out, 0);
        let pha_out_token = args.request_fetch(&pha_out, 1);
        let ir1_out_token = args.request_fetch(&ir1_out, 2);
        let ir2_out_token = args.request_fetch(&ir2_out, 3);
        let ir3_out_token = args.request_fetch(&ir3_out, 4);
        let ir4_out_token = args.request_fetch(&ir4_out, 5);

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

        Ok((&*fgr).into())
    }
}
