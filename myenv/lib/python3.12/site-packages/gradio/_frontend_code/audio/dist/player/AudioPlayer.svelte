<script>import { onMount } from "svelte";
import { Music } from "@gradio/icons";
import { format_time } from "@gradio/utils";
import WaveSurfer from "wavesurfer.js";
import { skip_audio, process_audio } from "../shared/utils";
import WaveformControls from "../shared/WaveformControls.svelte";
import { Empty } from "@gradio/atoms";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
import { createEventDispatcher } from "svelte";
import Hls from "hls.js";
export let value = null;
$:
  url = value?.url;
export let label;
export let i18n;
export let dispatch_blob = () => Promise.resolve();
export let interactive = false;
export let editable = true;
export let trim_region_settings = {};
export let waveform_settings;
export let waveform_options;
export let mode = "";
export let loop;
export let handle_reset_value = () => {
};
let container;
let waveform;
let playing = false;
let timeRef;
let durationRef;
let audio_duration;
let trimDuration = 0;
let show_volume_slider = false;
let audio_player;
let stream_active = false;
const dispatch = createEventDispatcher();
$:
  use_waveform = waveform_options.show_recording_waveform && !value?.is_stream;
const create_waveform = () => {
  waveform = WaveSurfer.create({
    container,
    ...waveform_settings
  });
  resolve_wasm_src(value?.url).then((resolved_src) => {
    if (resolved_src && waveform) {
      return waveform.load(resolved_src);
    }
  });
};
$:
  if (use_waveform && container !== void 0 && container !== null) {
    if (waveform !== void 0)
      waveform.destroy();
    container.innerHTML = "";
    create_waveform();
    playing = false;
  }
$:
  waveform?.on("decode", (duration) => {
    audio_duration = duration;
    durationRef && (durationRef.textContent = format_time(duration));
  });
$:
  waveform?.on(
    "timeupdate",
    (currentTime) => timeRef && (timeRef.textContent = format_time(currentTime))
  );
$:
  waveform?.on("ready", () => {
    if (!waveform_settings.autoplay) {
      waveform?.stop();
    } else {
      waveform?.play();
    }
  });
$:
  waveform?.on("finish", () => {
    if (loop) {
      waveform?.play();
    } else {
      playing = false;
      dispatch("stop");
    }
  });
$:
  waveform?.on("pause", () => {
    playing = false;
    dispatch("pause");
  });
$:
  waveform?.on("play", () => {
    playing = true;
    dispatch("play");
  });
$:
  waveform?.on("load", () => {
    dispatch("load");
  });
const handle_trim_audio = async (start, end) => {
  mode = "";
  const decodedData = waveform?.getDecodedData();
  if (decodedData)
    await process_audio(
      decodedData,
      start,
      end,
      waveform_settings.sampleRate
    ).then(async (trimmedBlob) => {
      await dispatch_blob([trimmedBlob], "change");
      waveform?.destroy();
      container.innerHTML = "";
    });
  dispatch("edit");
};
async function load_audio(data) {
  stream_active = false;
  await resolve_wasm_src(data).then((resolved_src) => {
    if (!resolved_src || value?.is_stream)
      return;
    if (waveform_options.show_recording_waveform) {
      waveform?.load(resolved_src);
    } else if (audio_player) {
      audio_player.src = resolved_src;
    }
  });
}
$:
  url && load_audio(url);
function load_stream(value2) {
  if (!value2 || !value2.is_stream || !value2.url)
    return;
  if (!audio_player)
    return;
  if (Hls.isSupported() && !stream_active) {
    const hls = new Hls({
      maxBufferLength: 1,
      maxMaxBufferLength: 1,
      lowLatencyMode: true
    });
    hls.loadSource(value2.url);
    hls.attachMedia(audio_player);
    hls.on(Hls.Events.MANIFEST_PARSED, function() {
      if (waveform_settings.autoplay)
        audio_player.play();
    });
    hls.on(Hls.Events.ERROR, function(event, data) {
      console.error("HLS error:", event, data);
      if (data.fatal) {
        switch (data.type) {
          case Hls.ErrorTypes.NETWORK_ERROR:
            console.error(
              "Fatal network error encountered, trying to recover"
            );
            hls.startLoad();
            break;
          case Hls.ErrorTypes.MEDIA_ERROR:
            console.error("Fatal media error encountered, trying to recover");
            hls.recoverMediaError();
            break;
          default:
            console.error("Fatal error, cannot recover");
            hls.destroy();
            break;
        }
      }
    });
    stream_active = true;
  } else if (!stream_active) {
    audio_player.src = value2.url;
    if (waveform_settings.autoplay)
      audio_player.play();
    stream_active = true;
  }
}
$:
  load_stream(value);
onMount(() => {
  window.addEventListener("keydown", (e) => {
    if (!waveform || show_volume_slider)
      return;
    if (e.key === "ArrowRight" && mode !== "edit") {
      skip_audio(waveform, 0.1);
    } else if (e.key === "ArrowLeft" && mode !== "edit") {
      skip_audio(waveform, -0.1);
    }
  });
});
</script>

<audio
	class="standard-player"
	class:hidden={use_waveform}
	controls
	autoplay={waveform_settings.autoplay}
	on:load
	bind:this={audio_player}
	on:ended={() => dispatch("stop")}
	on:play={() => dispatch("play")}
/>
{#if value === null}
	<Empty size="small">
		<Music />
	</Empty>
{:else if use_waveform}
	<div
		class="component-wrapper"
		data-testid={label ? "waveform-" + label : "unlabelled-audio"}
	>
		<div class="waveform-container">
			<div
				id="waveform"
				bind:this={container}
				style:height={container ? null : "58px"}
			/>
		</div>

		<div class="timestamps">
			<time bind:this={timeRef} id="time">0:00</time>
			<div>
				{#if mode === "edit" && trimDuration > 0}
					<time id="trim-duration">{format_time(trimDuration)}</time>
				{/if}
				<time bind:this={durationRef} id="duration">0:00</time>
			</div>
		</div>

		<WaveformControls
			{container}
			{waveform}
			{playing}
			{audio_duration}
			{i18n}
			{interactive}
			{handle_trim_audio}
			bind:mode
			bind:trimDuration
			bind:show_volume_slider
			show_redo={interactive}
			{handle_reset_value}
			{waveform_options}
			{trim_region_settings}
			{editable}
		/>
	</div>
{/if}

<style>
	.component-wrapper {
		padding: var(--size-3);
		width: 100%;
	}

	:global(::part(wrapper)) {
		margin-bottom: var(--size-2);
	}

	.timestamps {
		display: flex;
		justify-content: space-between;
		align-items: center;
		width: 100%;
		padding: var(--size-1) 0;
	}

	#time {
		color: var(--neutral-400);
	}

	#duration {
		color: var(--neutral-400);
	}

	#trim-duration {
		color: var(--color-accent);
		margin-right: var(--spacing-sm);
	}
	.waveform-container {
		display: flex;
		align-items: center;
		justify-content: center;
		width: var(--size-full);
	}

	#waveform {
		width: 100%;
		height: 100%;
		position: relative;
	}

	.standard-player {
		width: 100%;
		padding: var(--size-2);
	}

	.hidden {
		display: none;
	}
</style>
