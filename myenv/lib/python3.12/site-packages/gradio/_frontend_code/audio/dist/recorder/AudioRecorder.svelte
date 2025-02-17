<script>import { onMount } from "svelte";
import { createEventDispatcher } from "svelte";
import WaveSurfer from "wavesurfer.js";
import { skip_audio, process_audio } from "../shared/utils";
import WSRecord from "wavesurfer.js/dist/plugins/record.js";
import WaveformControls from "../shared/WaveformControls.svelte";
import WaveformRecordControls from "../shared/WaveformRecordControls.svelte";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.js";
import { format_time } from "@gradio/utils";
export let mode;
export let i18n;
export let dispatch_blob;
export let waveform_settings;
export let waveform_options = {
  show_recording_waveform: true
};
export let handle_reset_value;
export let editable = true;
export let recording = false;
let micWaveform;
let recordingWaveform;
let playing = false;
let recordingContainer;
let microphoneContainer;
let record;
let recordedAudio = null;
let timeRef;
let durationRef;
let audio_duration;
let seconds = 0;
let interval;
let timing = false;
let trimDuration = 0;
const start_interval = () => {
  clearInterval(interval);
  interval = setInterval(() => {
    seconds++;
  }, 1e3);
};
const dispatch = createEventDispatcher();
function record_start_callback() {
  start_interval();
  timing = true;
  dispatch("start_recording");
  if (waveform_options.show_recording_waveform) {
    let waveformCanvas = microphoneContainer;
    if (waveformCanvas)
      waveformCanvas.style.display = "block";
  }
}
async function record_end_callback(blob) {
  seconds = 0;
  timing = false;
  clearInterval(interval);
  try {
    const array_buffer = await blob.arrayBuffer();
    const context = new AudioContext({
      sampleRate: waveform_settings.sampleRate
    });
    const audio_buffer = await context.decodeAudioData(array_buffer);
    if (audio_buffer)
      await process_audio(audio_buffer).then(async (audio) => {
        await dispatch_blob([audio], "change");
        await dispatch_blob([audio], "stop_recording");
      });
  } catch (e) {
    console.error(e);
  }
}
$:
  record?.on("record-resume", () => {
    start_interval();
  });
$:
  recordingWaveform?.on("decode", (duration) => {
    audio_duration = duration;
    durationRef && (durationRef.textContent = format_time(duration));
  });
$:
  recordingWaveform?.on(
    "timeupdate",
    (currentTime) => timeRef && (timeRef.textContent = format_time(currentTime))
  );
$:
  recordingWaveform?.on("pause", () => {
    dispatch("pause");
    playing = false;
  });
$:
  recordingWaveform?.on("play", () => {
    dispatch("play");
    playing = true;
  });
$:
  recordingWaveform?.on("finish", () => {
    dispatch("stop");
    playing = false;
  });
const create_mic_waveform = () => {
  if (microphoneContainer)
    microphoneContainer.innerHTML = "";
  if (micWaveform !== void 0)
    micWaveform.destroy();
  if (!microphoneContainer)
    return;
  micWaveform = WaveSurfer.create({
    ...waveform_settings,
    normalize: false,
    container: microphoneContainer
  });
  record = micWaveform.registerPlugin(RecordPlugin.create());
  record?.on("record-end", record_end_callback);
  record?.on("record-start", record_start_callback);
  record?.on("record-pause", () => {
    dispatch("pause_recording");
    clearInterval(interval);
  });
  record?.on("record-end", (blob) => {
    recordedAudio = URL.createObjectURL(blob);
    const microphone = microphoneContainer;
    const recording2 = recordingContainer;
    if (microphone)
      microphone.style.display = "none";
    if (recording2 && recordedAudio) {
      recording2.innerHTML = "";
      create_recording_waveform();
    }
  });
};
const create_recording_waveform = () => {
  let recording2 = recordingContainer;
  if (!recordedAudio || !recording2)
    return;
  recordingWaveform = WaveSurfer.create({
    container: recording2,
    url: recordedAudio,
    ...waveform_settings
  });
};
const handle_trim_audio = async (start, end) => {
  mode = "edit";
  const decodedData = recordingWaveform.getDecodedData();
  if (decodedData)
    await process_audio(decodedData, start, end).then(
      async (trimmedAudio) => {
        await dispatch_blob([trimmedAudio], "change");
        await dispatch_blob([trimmedAudio], "stop_recording");
        recordingWaveform.destroy();
        create_recording_waveform();
      }
    );
  dispatch("edit");
};
onMount(() => {
  create_mic_waveform();
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") {
      skip_audio(recordingWaveform, 0.1);
    } else if (e.key === "ArrowLeft") {
      skip_audio(recordingWaveform, -0.1);
    }
  });
});
</script>

<div class="component-wrapper">
	<div
		class="microphone"
		bind:this={microphoneContainer}
		data-testid="microphone-waveform"
	/>
	<div bind:this={recordingContainer} data-testid="recording-waveform" />

	{#if (timing || recordedAudio) && waveform_options.show_recording_waveform}
		<div class="timestamps">
			<time bind:this={timeRef} class="time">0:00</time>
			<div>
				{#if mode === "edit" && trimDuration > 0}
					<time class="trim-duration">{format_time(trimDuration)}</time>
				{/if}
				{#if timing}
					<time class="duration">{format_time(seconds)}</time>
				{:else}
					<time bind:this={durationRef} class="duration">0:00</time>
				{/if}
			</div>
		</div>
	{/if}

	{#if microphoneContainer && !recordedAudio}
		<WaveformRecordControls
			bind:record
			{i18n}
			{timing}
			{recording}
			show_recording_waveform={waveform_options.show_recording_waveform}
			record_time={format_time(seconds)}
		/>
	{/if}

	{#if recordingWaveform && recordedAudio}
		<WaveformControls
			bind:waveform={recordingWaveform}
			container={recordingContainer}
			{playing}
			{audio_duration}
			{i18n}
			{editable}
			interactive={true}
			{handle_trim_audio}
			bind:trimDuration
			bind:mode
			show_redo
			{handle_reset_value}
			{waveform_options}
		/>
	{/if}
</div>

<style>
	.microphone {
		width: 100%;
		display: none;
	}

	.component-wrapper {
		padding: var(--size-3);
		width: 100%;
	}

	.timestamps {
		display: flex;
		justify-content: space-between;
		align-items: center;
		width: 100%;
		padding: var(--size-1) 0;
		margin: var(--spacing-md) 0;
	}

	.time {
		color: var(--neutral-400);
	}

	.duration {
		color: var(--neutral-400);
	}

	.trim-duration {
		color: var(--color-accent);
		margin-right: var(--spacing-sm);
	}
</style>
