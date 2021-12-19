#[derive(Debug, Default)]
pub struct InitialRecurrentState {
    r1: tch::jit::IValue::Double,
    r2: tch::jit::IValue::Double,
    r3: tch::jit::IValue::Double,
    r4: tch::jit::IValue::Double,
}

#[derive(Debug)]
pub struct RobustVideoMatting {
    module: tch::CModule,
    state: InitialRecurrentState,
}
impl RobustVideoMatting {
    pub fn try_init() -> HideResult<Self> {
        let mut module = tch::CModule::load("models/rvm_mobilenetv3_fp32.torchscript")?;
        module.set_eval();

        Ok(Self {
            module,
            state: Default::default(),
        })
    }

    pub fn run(&self, frame: ImageBuffer) -> HideResult<()> {
        self.module.forward_ts(&[])?;
    }
}
