# hide-my-mess

Utility to have virtual backgrounds on your webcam. Think Google Meet, Zoom etc virtual backgrounds feature but better in a sense.

## About

Better is a bold claim you might say, but it uses [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) as an inference algorithm.
This algorithm is a bit special since unlike any other algorithm, it uses recurrent neural networks with a temporal memory over past frames, which makes it way better at detecting what is and what is not a human.

I started building this out of a need for business meetings.
I run Linux for work, and most of the landscape for those solutions are using poor algorithms and/or are written in inefficient languages (**cough** *python*) for this very purpose.

Building instructions can be found below. Uploaded builds will be available once I find some time to setup the CI.

## Technology

**Only compatible with Linux as of now.**
I originally intended to make it multiplatform but creating virtual cameras is a much more involved endeavor on macOS and Windows.

Written in Rust, uses Tensorflow for the ML Inference and as mentioned before, RobustVideoMatting as an inference model.

For the virtual camera itself, it relies on the presence of `v4l2loopback` called with the correct parameters.

## Prequisites

* `v4l2loopback`, installed, and with a device labeled `"fake-cam"`

Example: `sudo modprobe v4l2loopback devices=1 exclusive_caps=1 max_buffers=2 video_nr=2 card_label="fake-cam"`

**Important: If you wish to use your virtual webcam with Chrome, browsers, or any other electron-based client (Slack, Discord, etc), the `exclusive_caps=1` parameter is absolutely needed!**

`hide-my-mess` will download the requested RVM model on first launch and store it in `~/.cache/hide-my-mess/models`.

By default, it runs inference on the GPU. You can decide to run inference on the CPU by compiling the program with `--no-default-features`.

## Usage

```
hide-my-mess 0.1.0
Mathieu Amiot <amiot.mathieu@gmail.com>
Virtual camera reliably crops your wonderful self out of your webcam

USAGE:
    hide-my-mess [OPTIONS]

OPTIONS:
        --bg <BACKGROUND>
            Replaces the inferred background by the provided image path

        --blur
            Dynamic background blurring. Be wary that this has a big cost in performance Is also
            incompatible with the --background option

        --camera-index <CAMERA_INDEX>
            Choose a video capture device index manually, in case you have several connected

        --fps <FPS>
            Desired capture framerate. If not available, it'll fallback to the highest mode detected

    -h, --height <HEIGHT>
            Desired frame capture height. If not available, it'll fallback to the highest mode
            detected

        --help
            Print help information

    -l, --list
            Lists video capture devices present on the system along with their details

        --model <MODEL>
            Used variant of RobustVideoMatting. resnet is more accurate & faster but heavier,
            mobilenet is lighter The two possible choices are `resnet50` and `mobilenetv3` [default:
            mobilenetv3]

    -V, --version
            Print version information

    -w, --width <WIDTH>
            Desired frame capture width. If not available, it'll fallback to the highest mode
            detected
```

## Building & Contributing

Requirements:

* Rust
* `v4l2` & `v4l2loopback`

Then:

* GPU Inference: `cargo build --release`
* CPU Inference: `cargo build --no-default-features --release`

That's it!

## License

Licensed under either of these:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
   [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

## Authors

Mathieu "@OtaK_" Amiot
