<script>import { onDestroy, createEventDispatcher, tick } from "svelte";
import { Upload, ModifyUpload } from "@gradio/upload";
import { prepare_files } from "@gradio/client";
import { BlockLabel } from "@gradio/atoms";
import { Music } from "@gradio/icons";
import { StreamingBar } from "@gradio/statustracker";
import AudioPlayer from "../player/AudioPlayer.svelte";
import AudioRecorder from "../recorder/AudioRecorder.svelte";
import StreamAudio from "../streaming/StreamAudio.svelte";
import { SelectSource } from "@gradio/atoms";
export let value = null;
export let label;
export let root;
export let loop;
export let show_label = true;
export let show_download_button = false;
export let sources = ["microphone", "upload"];
export let pending = false;
export let streaming = false;
export let i18n;
export let waveform_settings;
export let trim_region_settings = {};
export let waveform_options = {};
export let dragging;
export let active_source;
export let handle_reset_value = () => {
};
export let editable = true;
export let max_file_size = null;
export let upload;
export let stream_handler;
export let stream_every;
export let uploading = false;
export let recording = false;
export let class_name = "";
let time_limit = null;
let stream_state = "closed";
export const modify_stream = (state) => {
  if (state === "closed") {
    time_limit = null;
    stream_state = "closed";
  } else if (state === "waiting") {
    stream_state = "waiting";
  } else {
    stream_state = "open";
  }
};
export const set_time_limit = (time) => {
  if (recording)
    time_limit = time;
};
$:
  dispatch("drag", dragging);
let recorder;
let mode = "";
let header = void 0;
let pending_stream = [];
let submit_pending_stream_on_pending_end = false;
let inited = false;
const NUM_HEADER_BYTES = 44;
let audio_chunks = [];
let module_promises;
function get_modules() {
  module_promises = [
    import("extendable-media-recorder"),
    import("extendable-media-recorder-wav-encoder")
  ];
}
const is_browser = typeof window !== "undefined";
if (is_browser && streaming) {
  get_modules();
}
const dispatch = createEventDispatcher();
const dispatch_blob = async (blobs, event) => {
  let _audio_blob = new File(blobs, "audio.wav");
  const val = await prepare_files([_audio_blob], event === "stream");
  value = ((await upload(val, root, void 0, max_file_size || void 0))?.filter(
    Boolean
  ))[0];
  dispatch(event, value);
};
onDestroy(() => {
  if (streaming && recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
});
async function prepare_audio() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (!navigator.mediaDevices) {
      dispatch("error", i18n("audio.no_device_support"));
      return;
    }
    if (err instanceof DOMException && err.name == "NotAllowedError") {
      dispatch("error", i18n("audio.allow_recording_access"));
      return;
    }
    throw err;
  }
  if (stream == null)
    return;
  if (streaming) {
    const [{ MediaRecorder: MediaRecorder2, register }, { connect }] = await Promise.all(module_promises);
    await register(await connect());
    recorder = new MediaRecorder2(stream, { mimeType: "audio/wav" });
    recorder.addEventListener("dataavailable", handle_chunk);
  } else {
    recorder = new MediaRecorder(stream);
    recorder.addEventListener("dataavailable", (event) => {
      audio_chunks.push(event.data);
    });
  }
  recorder.addEventListener("stop", async () => {
    recording = false;
    await dispatch_blob(audio_chunks, "change");
    await dispatch_blob(audio_chunks, "stop_recording");
    audio_chunks = [];
  });
  inited = true;
}
async function handle_chunk(event) {
  let buffer = await event.data.arrayBuffer();
  let payload = new Uint8Array(buffer);
  if (!header) {
    header = new Uint8Array(buffer.slice(0, NUM_HEADER_BYTES));
    payload = new Uint8Array(buffer.slice(NUM_HEADER_BYTES));
  }
  if (pending) {
    pending_stream.push(payload);
  } else {
    let blobParts = [header].concat(pending_stream, [payload]);
    if (!recording || stream_state === "waiting")
      return;
    dispatch_blob(blobParts, "stream");
    pending_stream = [];
  }
}
$:
  if (submit_pending_stream_on_pending_end && pending === false) {
    submit_pending_stream_on_pending_end = false;
    if (header && pending_stream) {
      let blobParts = [header].concat(pending_stream);
      pending_stream = [];
      dispatch_blob(blobParts, "stream");
    }
  }
async function record() {
  recording = true;
  dispatch("start_recording");
  if (!inited)
    await prepare_audio();
  header = void 0;
  if (streaming && recorder.state != "recording") {
    recorder.start(stream_every * 1e3);
  }
}
function clear() {
  dispatch("change", null);
  dispatch("clear");
  mode = "";
  value = null;
}
function handle_load({ detail }) {
  value = detail;
  dispatch("change", detail);
  dispatch("upload", detail);
}
async function stop() {
  recording = false;
  if (streaming) {
    dispatch("close_stream");
    dispatch("stop_recording");
    recorder.stop();
    if (pending) {
      submit_pending_stream_on_pending_end = true;
    }
    dispatch_blob(audio_chunks, "stop_recording");
    dispatch("clear");
    mode = "";
  }
}
$:
  if (!recording && recorder)
    stop();
$:
  if (recording && recorder)
    record();
</script>

<BlockLabel
	{show_label}
	Icon={Music}
	float={active_source === "upload" && value === null}
	label={label || i18n("audio.audio")}
/>
<div class="audio-container {class_name}">
	<StreamingBar {time_limit} />
	{#if value === null || streaming}
		{#if active_source === "microphone"}
			<ModifyUpload {i18n} on:clear={clear} />
			{#if streaming}
				<StreamAudio
					{record}
					{recording}
					{stop}
					{i18n}
					{waveform_settings}
					{waveform_options}
					waiting={stream_state === "waiting"}
				/>
			{:else}
				<AudioRecorder
					bind:mode
					{i18n}
					{editable}
					{recording}
					{dispatch_blob}
					{waveform_settings}
					{waveform_options}
					{handle_reset_value}
					on:start_recording
					on:pause_recording
					on:stop_recording
				/>
			{/if}
		{:else if active_source === "upload"}
			<!-- explicitly listed out audio mimetypes due to iOS bug not recognizing audio/* -->
			<Upload
				filetype="audio/aac,audio/midi,audio/mpeg,audio/ogg,audio/wav,audio/x-wav,audio/opus,audio/webm,audio/flac,audio/vnd.rn-realaudio,audio/x-ms-wma,audio/x-aiff,audio/amr,audio/*"
				on:load={handle_load}
				bind:dragging
				bind:uploading
				on:error={({ detail }) => dispatch("error", detail)}
				{root}
				{max_file_size}
				{upload}
				{stream_handler}
			>
				<slot />
			</Upload>
		{/if}
	{:else}
		<ModifyUpload
			{i18n}
			on:clear={clear}
			on:edit={() => (mode = "edit")}
			download={show_download_button ? value.url : null}
		/>

		<AudioPlayer
			bind:mode
			{value}
			{label}
			{i18n}
			{dispatch_blob}
			{waveform_settings}
			{waveform_options}
			{trim_region_settings}
			{handle_reset_value}
			{editable}
			{loop}
			interactive
			on:stop
			on:play
			on:pause
			on:edit
		/>
	{/if}
	<SelectSource {sources} bind:active_source handle_clear={clear} />
</div>

<style>
	.audio-container {
		height: calc(var(--size-full) - var(--size-6));
		display: flex;
		flex-direction: column;
		justify-content: space-between;
	}

	.audio-container.compact-audio {
		margin-top: calc(var(--size-8) * -1);
		height: auto;
		padding: 0px;
		gap: var(--size-2);
		min-height: var(--size-5);
	}

	.compact-audio :global(.audio-player) {
		padding: 0px;
	}

	.compact-audio :global(.controls) {
		gap: 0px;
		padding: 0px;
	}

	.compact-audio :global(.waveform-container) {
		height: var(--size-12) !important;
	}

	.compact-audio :global(.player-container) {
		min-height: unset;
		height: auto;
	}
</style>
