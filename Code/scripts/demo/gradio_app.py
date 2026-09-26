"""Gradio 6 application for the current Ex-Omni 2D pipeline."""

from __future__ import annotations

import argparse
import html
from urllib.parse import quote
import atexit
import os
from pathlib import Path
import sys
import traceback
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo.runtime import (
    DemoRuntime,
    StreamingGPUResourceError,
    SwitchableDemoRuntime,
    default_config_path,
    is_gpu_resource_error,
    require_streaming_split_gpus,
    select_single_gpu_id,
)

STREAMING_DEPLOYMENT_ENV = "EX_OMNI_DEMO_STREAMING_DEPLOYMENT"
STREAMING_SINGLE_GPU_ENV = "EX_OMNI_DEMO_STREAMING_GPU"
DEFAULT_ROLE_STATUS = (
    "Default reference image and reference voice loaded."
)
MODEL_CHOICES = [
    ("Full-sequence (Quality)", "full_sequence"),
    ("Streaming (Fast)", "streaming"),
]
STREAMING_DIFFUSION_STEPS = [8]
FULL_SEQUENCE_DIFFUSION_STEPS = [50]
VIDEO_RESET_JS = (Path(__file__).with_name('idle_player.js')).read_text()
RESET_REPLY_JS = '(...args) => { window.exOmniResetReply?.(); return args; }'
INITIALIZE_AVATAR_JS = """(...args) => {
    window.exOmniResetReply?.();
    const status = document.querySelector('#avatar-status textarea');
    if (status) status.value = '初始化中 · Preparing the avatar standby animation...';
    return args;
}"""



ROLE_CONFIGURATION_CSS = """
#avatar-stage { position: relative; height: 380px; background: #111; overflow: hidden; }
#avatar-stage video, #avatar-stage canvas { position: absolute; inset: 0;
    width: 100%; height: 100%; object-fit: contain; }
#avatar-stage [hidden] { display: none !important; }
#avatar-idle-source, #avatar-reply-source { display: none !important; }
.avatar-play-button { position: absolute; bottom: 24px; left: 30%; z-index: 10;
    background: #2563eb; color: white; padding: 10px 18px; border-radius: 8px; cursor: pointer; }
.role-config-item {
    height: 220px !important;
    min-height: 220px !important;
}
.role-config-text textarea {
    height: 158px !important;
    min-height: 158px !important;
    overflow-y: auto !important;
    scrollbar-gutter: stable;
    resize: none !important;
}
.role-config-audio .audio-container {
    height: 180px !important;
    min-height: 180px !important;
    max-height: 180px !important;
}
.status-box textarea {
    height: 60px !important;
    min-height: 60px !important;
}
"""


def video_payload(path, *, complete: bool = False) -> str:
    """Pass a local media URL to the stable HTML player without replacing it."""
    url = "/gradio_api/file=" + quote(str(Path(path).resolve()), safe="/")
    return '<span data-video-src="' + html.escape(url, quote=True) + '" data-video-complete="' + str(int(complete)) + '"></span>'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Ex-Omni 2D Gradio application."
    )
    parser.add_argument(
        "--mode",
        choices=("streaming", "full_sequence"),
        default="full_sequence",
        help="Full-sequence is the default quality model; streaming uses the eight-step checkpoint.",
    )
    parser.add_argument("--config", help="Release YAML; defaults depend on --mode.")
    parser.add_argument("--streaming-config", dest="streaming_config", help="Streaming model YAML.")
    parser.add_argument("--full-sequence-config", dest="full_sequence_config", help="Full-sequence model YAML.")
    parser.add_argument("--output-dir", default="outputs/demo/gradio")
    parser.add_argument("--full-sequence-gpu", dest="full_sequence_gpu", type=int, default=0)
    parser.add_argument("--service-timeout-seconds", type=float, default=1200)
    parser.add_argument(
        "--streaming-deployment",
        dest="streaming_deployment",
        choices=("auto", "split", "single_gpu"),
        default=os.environ.get(STREAMING_DEPLOYMENT_ENV, "auto"),
        help="Use the configured split topology; auto allows single-GPU resource fallback.",
    )
    parser.add_argument(
        "--streaming-gpu",
        dest="single_gpu",
        type=int,
        default=(
            int(os.environ[STREAMING_SINGLE_GPU_ENV])
            if os.environ.get(STREAMING_SINGLE_GPU_ENV)
            else None
        ),
        help="Physical GPU used by the single-GPU streaming fallback.",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "8080"))))
    parser.add_argument("--share", action="store_true")
    return parser


def _replace_cli_option(
    argv: list[str],
    option: str,
    value: str,
) -> list[str]:
    result = list(argv)
    try:
        index = result.index(option)
    except ValueError:
        result.extend([option, value])
        return result
    if index + 1 < len(result):
        result[index + 1] = value
    else:
        result.append(value)
    return result


def _restart_with_single_gpu(args, streaming_config: Path, reason: BaseException):
    gpu_id = select_single_gpu_id(
        streaming_config,
        requested_gpu=args.single_gpu,
    )
    print(
        "[streaming] Split-pipeline startup could not acquire GPU resources; "
        f"restarting on single GPU {gpu_id}. Reason: "
        f"{type(reason).__name__}: {reason}",
        file=sys.stderr,
        flush=True,
    )
    child_argv = _replace_cli_option(
        sys.argv,
        "--streaming-deployment",
        "single_gpu",
    )
    child_argv = _replace_cli_option(
        child_argv,
        "--streaming-gpu",
        str(gpu_id),
    )
    environment = dict(os.environ)
    environment[STREAMING_DEPLOYMENT_ENV] = "single_gpu"
    environment[STREAMING_SINGLE_GPU_ENV] = str(gpu_id)
    environment["PYTHONUNBUFFERED"] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, *child_argv],
        environment,
    )


def create_app(runtime: DemoRuntime | SwitchableDemoRuntime):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gradio is not installed. Install the demo dependencies with "
            "`pip install -e '.[runtime,distributed,demo]'`."
        ) from exc

    def submit(
        text,
        speech_input,
        ref_image,
        ref_audio,
        model_mode,
        temperature,
        top_p,
        max_new_tokens,
        diffusion_steps,
        text_cfg,
        audio_cfg,
        history,
        session_id,
        avatar_ready,
    ):
        from scripts.demo.idle import image_identity

        if not ref_image or not avatar_ready or avatar_ready.get("identity") != image_identity(ref_image, str(model_mode).lower()):
            yield (list(history or []), text, speech_input, "", None, None, "Please initialize the avatar before starting a conversation.", session_id)
            return
        prompt = str(text or "").strip()
        messages = list(history or [])
        requested_mode = str(model_mode).lower()
        if runtime.mode != requested_mode:
            switch_mode = getattr(runtime, "switch_mode", None)
            if not callable(switch_mode):
                yield (
                    messages,
                    text,
                    speech_input,
                    None,
                    None,
                    None,
                    f"{requested_mode.title()} model is not available.",
                    session_id,
                )
                return
            try:
                switch_mode(requested_mode)
            except Exception as exc:
                yield (
                    messages,
                    text,
                    speech_input,
                    None,
                    None,
                    None,
                    f"Failed to load {requested_mode.title()}: "
                    f"{type(exc).__name__}: {exc}",
                    session_id,
                )
                return
        if not prompt and not speech_input:
            yield (
                messages,
                text,
                speech_input,
                None,
                None,
                None,
                "Enter text or provide speech input.",
                session_id,
            )
            return
        if not ref_image or not ref_audio:
            yield (
                messages,
                text,
                speech_input,
                None,
                None,
                None,
                "Reference image and reference audio are required.",
                session_id,
            )
            return
        if prompt and speech_input:
            user_content = [
                prompt,
                gr.Audio(value=speech_input, autoplay=False),
            ]
        elif speech_input:
            user_content = gr.Audio(value=speech_input, autoplay=False)
        else:
            user_content = prompt
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ""})
        yield (
            messages,
            "",
            None,
            "",
            None,
            None,
            "Generating results...",
            session_id,
        )
        try:
            for update in runtime.run_turn(
                prompt,
                session_id=session_id,
                ref_image=ref_image,
                ref_audio=ref_audio,
                role_card=None,
                speech_file=speech_input,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max(int(max_new_tokens), 512),
                diffusion_steps=8 if requested_mode == "streaming" else int(diffusion_steps),
                text_cfg=float(text_cfg),
                audio_cfg=float(audio_cfg),
            ):
                if update.response_text:
                    messages[-1] = {
                        "role": "assistant",
                        "content": update.response_text,
                    }
                if update.video_chunk_path is not None:
                    video = video_payload(update.video_chunk_path, complete=update.complete)
                elif update.video_path is not None:
                    video = video_payload(update.video_path, complete=update.complete)
                else:
                    video = gr.skip()
                if update.speech_path is not None:
                    speech = str(update.speech_path)
                else:
                    speech = gr.skip()
                yield (
                    messages,
                    "",
                    None,
                    update.vtp or "",
                    speech,
                    video,
                    update.status,
                    session_id,
                )
        except Exception as exc:
            traceback.print_exc()
            messages[-1] = {
                "role": "assistant",
                "content": f"Error: {type(exc).__name__}: {exc}",
            }
            yield (
                messages,
                "",
                None,
                "",
                None,
                None,
                f"Inference failed: {type(exc).__name__}: {exc}",
                session_id,
            )

    def initialize_avatar(ref_image, model_mode, previous_session_id=None):
        from scripts.demo.idle import image_identity

        new_session_id = f"web-{uuid.uuid4().hex[:10]}"
        cleared = ([], None, "", None, new_session_id)
        yield None, "初始化中 · Preparing the avatar standby animation. Please wait...", gr.update(interactive=False), None, *cleared
        try:
            if previous_session_id:
                runtime.clear_session(previous_session_id)
            if not ref_image:
                raise ValueError("Please upload a reference image first.")
            if runtime.mode != model_mode:
                runtime.switch_mode(model_mode)
            path = runtime.prepare_idle(ref_image)
            ready = {"identity": image_identity(ref_image, runtime.mode), "path": str(path)}
            yield video_payload(path), "Initialization complete. You can start a conversation.", gr.update(interactive=True), ready, *cleared
        except Exception as exc:
            traceback.print_exc()
            yield None, f"Could not prepare the standby animation. Click Reinitialize Avatar to retry: {type(exc).__name__}: {exc}", gr.update(interactive=False), None, *cleared

    def diffusion_step_update(mode: str, *, loading: bool = False):
        if mode == "full_sequence":
            return gr.update(
                choices=FULL_SEQUENCE_DIFFUSION_STEPS,
                value=50,
                interactive=False,
                visible=True,
            )
        return gr.update(
            choices=STREAMING_DIFFUSION_STEPS,
            value=8,
            interactive=False,
            visible=False,
        )

    def guidance_update(
        mode: str,
        *,
        value: float | None = None,
        loading: bool = False,
    ):
        if value is None:
            value = 4.0 if mode == "full_sequence" else 1.0
        return gr.update(
            value=float(value),
            interactive=mode == "full_sequence" and not loading,
        )

    def select_model(model_mode, session_id):
        requested_mode = str(model_mode).lower()
        yield (
            f"初始化中 · Loading {dict((value, label) for label, value in MODEL_CHOICES)[requested_mode]}...",
            [],
            None,
            None,
            "",
            session_id,
            generation_defaults["temperature"],
            generation_defaults["top_p"],
            generation_defaults["max_new_tokens"],
            diffusion_step_update(requested_mode, loading=True),
            guidance_update(requested_mode, loading=True),
            guidance_update(requested_mode, loading=True),
            gr.skip(),
        )
        try:
            switch_mode = getattr(runtime, "switch_mode", None)
            if not callable(switch_mode):
                raise RuntimeError("This runtime does not support model switching")
            switch_mode(requested_mode)
            defaults = runtime.generation_defaults
            yield (
                "初始化中 · Model loaded; preparing the avatar...",
                [],
                None,
                None,
                "",
                f"web-{uuid.uuid4().hex[:10]}",
                defaults["temperature"],
                defaults["top_p"],
                defaults["max_new_tokens"],
                diffusion_step_update(requested_mode),
                guidance_update(
                    requested_mode,
                    value=defaults["text_cfg"],
                ),
                guidance_update(
                    requested_mode,
                    value=defaults["audio_cfg"],
                ),
                gr.skip(),
            )
        except Exception as exc:
            current_mode = runtime.mode
            defaults = runtime.generation_defaults
            if (
                requested_mode == "streaming"
                and type(exc).__name__ == "RemoteWeightUnavailableError"
            ):
                failure = (
                    "Streaming weights are not available in the Hugging Face "
                    "release yet. Continuing with the full-sequence model."
                )
            else:
                failure = (
                    f"Failed to load {requested_mode.title()}: "
                    f"{type(exc).__name__}: {exc}"
                )
            yield (
                failure,
                [],
                None,
                None,
                "",
                session_id,
                defaults["temperature"],
                defaults["top_p"],
                defaults["max_new_tokens"],
                diffusion_step_update(current_mode),
                guidance_update(
                    current_mode,
                    value=defaults["text_cfg"],
                ),
                guidance_update(
                    current_mode,
                    value=defaults["audio_cfg"],
                ),
                gr.update(value=current_mode),
            )

    def clear(session_id):
        runtime.clear_session(session_id)
        return (
            [],
            "",
            None,
            "",
            None,
            None,
            "Session cleared.",
            f"web-{uuid.uuid4().hex[:10]}",
        )

    generation_defaults = getattr(
        runtime,
        "generation_defaults",
        {
            "temperature": 0.3,
            "top_p": 0.85,
            "max_new_tokens": 512,
            "diffusion_steps": 8,
            "text_cfg": 1.0,
            "audio_cfg": 1.0,
        },
    )
    with gr.Blocks(title="Ex-Omni 2D") as app:
        gr.Markdown(
            f"# Ex-Omni-2D\n"
        )
        session_id = gr.State(lambda: f"web-{uuid.uuid4().hex[:10]}")
        avatar_ready = gr.State(None)

        with gr.Accordion("Role configuration", open=True):
            gr.Markdown(
                "Reference image and reference voice are required for the first "
                "turn and reused for the rest of the session.\n\n"
                f"✅ **{DEFAULT_ROLE_STATUS}**"
            )
            with gr.Row(equal_height=True):
                vtp = gr.Textbox(
                    label="Visual Thought Plan (VTP)",
                    lines=5,
                    max_lines=5,
                    interactive=False,
                    scale=2,
                    elem_classes=["role-config-item", "role-config-text"],
                )
                ref_image = gr.Image(
                    label="Reference image",
                    type="filepath",
                    value=None,
                    height=180,
                    scale=1,
                    elem_classes="role-config-item",
                )
                ref_audio = gr.Audio(
                    label="Reference voice",
                    type="filepath",
                    sources=["upload", "microphone"],
                    value=None,
                    scale=1,
                    elem_classes=["role-config-item", "role-config-audio"],
                )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=460,
                    group_consecutive_messages=True,
                )
                with gr.Row(equal_height=True):
                    message = gr.Textbox(
                        label="Message",
                        elem_id="avatar-message",
                        placeholder="Ask the avatar something...",
                        lines=10,
                        max_lines=10,
                        scale=1,
                    )
                    speech_input = gr.Audio(
                        label="Speech Input",
                        type="filepath",
                        sources=["upload", "microphone"],
                        scale=1,
                    )
                with gr.Row():
                    send = gr.Button("Send", variant="primary", interactive=False)
                    clear_button = gr.Button("Clear")
            with gr.Column(scale=2):
                status = gr.Textbox(
                    label="Status",
                    elem_id="avatar-status",
                    value="初始化中 · Initializing the avatar...",
                    interactive=False,
                    elem_classes="status-box",
                )
                speech_output = gr.Audio(
                    label="Speech Output",
                    autoplay=False,
                    streaming=False,
                    interactive=False,
                )
                gr.HTML('<div id="avatar-stage" aria-label="Avatar player">'
                        '<video id="avatar-idle-video" muted loop playsinline></video>'
                        '<canvas class="avatar-reply-snapshot" hidden></canvas>'
                        '<video id="multimodal-response-video" playsinline hidden></video>'
                        '<button class="avatar-play-button" hidden>Play Response</button></div>')
                idle_video = gr.HTML(elem_id="avatar-idle-source")
                video = gr.HTML(elem_id="avatar-reply-source")
                initialize_button = gr.Button("Reinitialize Avatar")
                with gr.Accordion("Model configuration", open=True):
                    model_mode = gr.Radio(
                        label="Inference model",
                        choices=MODEL_CHOICES,
                        value=runtime.mode,
                    )
                    diffusion_steps = gr.Radio(
                        label="Diffusion steps",
                        choices=(
                            STREAMING_DIFFUSION_STEPS
                            if runtime.mode == "streaming"
                            else FULL_SEQUENCE_DIFFUSION_STEPS
                        ),
                        value=(
                            8
                            if runtime.mode == "streaming"
                            else 50
                        ),
                        interactive=False,
                        visible=runtime.mode == "full_sequence",
                    )
                    with gr.Row():
                        text_cfg = gr.Slider(
                            label="Text CFG",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=generation_defaults["text_cfg"],
                            interactive=runtime.mode == "full_sequence",
                        )
                        audio_cfg = gr.Slider(
                            label="Audio CFG",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=generation_defaults["audio_cfg"],
                            interactive=runtime.mode == "full_sequence",
                        )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=generation_defaults["temperature"],
                    )
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=generation_defaults["top_p"],
                    )
                    max_new_tokens = gr.Number(
                        label="Max new tokens",
                        minimum=512,
                        step=1,
                        precision=0,
                        value=generation_defaults["max_new_tokens"],
                    )
        inputs = [
            message,
            speech_input,
            ref_image,
            ref_audio,
            model_mode,
            temperature,
            top_p,
            max_new_tokens,
            diffusion_steps,
            text_cfg,
            audio_cfg,
            chatbot,
            session_id,
            avatar_ready,
        ]
        outputs = [
            chatbot,
            message,
            speech_input,
            vtp,
            speech_output,
            video,
            status,
            session_id,
        ]
        init_outputs = [idle_video, status, send, avatar_ready, chatbot, video, vtp, speech_output, session_id]
        init_inputs = [ref_image, model_mode, session_id]
        app.load(initialize_avatar, inputs=init_inputs, outputs=init_outputs, concurrency_id="avatar-model", js=INITIALIZE_AVATAR_JS)
        ref_image.change(initialize_avatar, inputs=init_inputs, outputs=init_outputs,
                         concurrency_id="avatar-model", js=INITIALIZE_AVATAR_JS)
        initialize_button.click(initialize_avatar, inputs=init_inputs, outputs=init_outputs,
                                concurrency_id="avatar-model", js=INITIALIZE_AVATAR_JS)
        send.click(submit, inputs=inputs, outputs=outputs, concurrency_id="avatar-model", js=RESET_REPLY_JS)
        message.submit(submit, inputs=inputs, outputs=outputs, concurrency_id="avatar-model", js=RESET_REPLY_JS)
        model_mode.change(
            select_model,
            inputs=[model_mode, session_id],
            outputs=[
                status,
                chatbot,
                speech_output,
                video,
                vtp,
                session_id,
                temperature,
                top_p,
                max_new_tokens,
                diffusion_steps,
                text_cfg,
                audio_cfg,
                model_mode,
            ],
            concurrency_id="avatar-model", js=RESET_REPLY_JS,
        ).then(initialize_avatar, inputs=init_inputs, outputs=init_outputs, concurrency_id="avatar-model", js=INITIALIZE_AVATAR_JS)
        clear_button.click(
            clear,
            inputs=[session_id],
            outputs=outputs, concurrency_id="avatar-model", js=RESET_REPLY_JS,
        )
    return app


def main() -> int:
    args = build_parser().parse_args()
    initial_config = (
        Path(args.config).expanduser()
        if args.config
        else default_config_path(args.mode)
    )
    streaming_config = (
        Path(args.streaming_config).expanduser()
        if args.streaming_config
        else initial_config
        if args.mode == "streaming"
        else default_config_path("streaming")
    )
    full_sequence_config = (
        Path(args.full_sequence_config).expanduser()
        if args.full_sequence_config
        else initial_config
        if args.mode == "full_sequence"
        else default_config_path("full_sequence")
    )
    if args.mode == "streaming" and args.streaming_deployment == "auto":
        try:
            gpu_ids = require_streaming_split_gpus(streaming_config)
            print(
                f"[streaming] Split-pipeline startup selected GPUs {list(gpu_ids)}.",
                flush=True,
            )
        except StreamingGPUResourceError as exc:
            _restart_with_single_gpu(args, streaming_config, exc)
    try:
        runtime = SwitchableDemoRuntime(
            initial_mode=args.mode,
            streaming_config=streaming_config,
            full_sequence_config=full_sequence_config,
            output_dir=args.output_dir,
            full_sequence_gpu=args.full_sequence_gpu,
            service_timeout_seconds=args.service_timeout_seconds,
            streaming_deployment=args.streaming_deployment,
            single_gpu_id=args.single_gpu,
        ).start()
    except Exception as exc:
        if (
            args.mode == "streaming"
            and args.streaming_deployment == "auto"
            and is_gpu_resource_error(exc)
        ):
            _restart_with_single_gpu(args, streaming_config, exc)
        print(
            f"Failed to start demo: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        f"{runtime.mode.title()} model loaded successfully; launching Gradio.",
        flush=True,
    )
    atexit.register(runtime.close)
    try:
        app = create_app(runtime)
        app.queue(default_concurrency_limit=1).launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            css=ROLE_CONFIGURATION_CSS,
            js=VIDEO_RESET_JS,
            ssr_mode=False,
            allowed_paths=[
                str(Path(args.output_dir).expanduser().resolve()),
                str((PROJECT_ROOT / "asset").resolve()),
            ],
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
