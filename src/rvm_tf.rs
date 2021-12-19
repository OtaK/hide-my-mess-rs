use crate::error::HideResult;

#[derive(Debug)]
pub struct InitialRecurrentState {
    r1: tensorflow::Tensor<f64>,
    r2: tensorflow::Tensor<f64>,
    r3: tensorflow::Tensor<f64>,
    r4: tensorflow::Tensor<f64>,
}

impl InitialRecurrentState {
    pub fn try_new() -> HideResult<Self> {
        let t = tensorflow::Tensor::new(&[0]).with_values(&[0.])?;
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
    graph: tensorflow::Graph,
    model: tensorflow::SavedModelBundle,
    state: InitialRecurrentState,
}

impl RobustVideoMatting {
    pub fn try_init() -> HideResult<Self> {
        let mut graph = tensorflow::Graph::new();

        let model = tensorflow::SavedModelBundle::load(
            &tensorflow::SessionOptions::new(),
            &[] as &[&str],
            &mut graph,
            "./models/rvm_resnet50_tf",
        )?;

        let state = InitialRecurrentState::try_new()?;

        Ok(Self {
            graph,
            model,
            state,
        })
    }

    pub fn run(&self) {
        let mut args = tensorflow::SessionRunArgs::new();
        //args.add_feed()
    }
}
