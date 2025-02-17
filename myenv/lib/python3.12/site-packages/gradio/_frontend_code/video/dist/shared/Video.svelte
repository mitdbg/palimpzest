<script>import { createEventDispatcher } from "svelte";
import { loaded } from "./utils";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
import Hls from "hls.js";
export let src = void 0;
export let muted = void 0;
export let playsinline = void 0;
export let preload = void 0;
export let autoplay = void 0;
export let controls = void 0;
export let currentTime = void 0;
export let duration = void 0;
export let paused = void 0;
export let node = void 0;
export let loop;
export let is_stream;
export let processingVideo = false;
let resolved_src;
let stream_active = false;
let latest_src;
$: {
  resolved_src = src;
  latest_src = src;
  const resolving_src = src;
  resolve_wasm_src(resolving_src).then((s) => {
    if (latest_src === resolving_src) {
      resolved_src = s;
    }
  });
}
const dispatch = createEventDispatcher();
function load_stream(src2, is_stream2, node2) {
  if (!src2 || !is_stream2)
    return;
  if (!node2)
    return;
  if (Hls.isSupported() && !stream_active) {
    const hls = new Hls({
      maxBufferLength: 1,
      // 0.5 seconds (500 ms)
      maxMaxBufferLength: 1,
      // Maximum max buffer length in seconds
      lowLatencyMode: true
      // Enable low latency mode
    });
    hls.loadSource(src2);
    hls.attachMedia(node2);
    hls.on(Hls.Events.MANIFEST_PARSED, function() {
      node2.play();
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
  }
}
$:
  src, stream_active = false;
$:
  load_stream(src, is_stream, node);
</script>

<!--
The spread operator with `$$props` or `$$restProps` can't be used here
to pass props from the parent component to the <video> element
because of its unexpected behavior: https://github.com/sveltejs/svelte/issues/7404
For example, if we add {...$$props} or {...$$restProps}, the boolean props aside it like `controls` will be compiled as string "true" or "false" on the actual DOM.
Then, even when `controls` is false, the compiled DOM would be `<video controls="false">` which is equivalent to `<video controls>` since the string "false" is even truthy.
-->
<div class:hidden={!processingVideo} class="overlay">
	<span class="load-wrap">
		<span class="loader" />
	</span>
</div>
<video
	src={resolved_src}
	{muted}
	{playsinline}
	{preload}
	{autoplay}
	{controls}
	{loop}
	on:loadeddata={dispatch.bind(null, "loadeddata")}
	on:click={dispatch.bind(null, "click")}
	on:play={dispatch.bind(null, "play")}
	on:pause={dispatch.bind(null, "pause")}
	on:ended={dispatch.bind(null, "ended")}
	on:mouseover={dispatch.bind(null, "mouseover")}
	on:mouseout={dispatch.bind(null, "mouseout")}
	on:focus={dispatch.bind(null, "focus")}
	on:blur={dispatch.bind(null, "blur")}
	on:loadstart
	on:loadeddata
	on:loadedmetadata
	bind:currentTime
	bind:duration
	bind:paused
	bind:this={node}
	use:loaded={{ autoplay: autoplay ?? false }}
	data-testid={$$props["data-testid"]}
	crossorigin="anonymous"
>
	<slot />
</video>

<style>
	.overlay {
		position: absolute;
		background-color: rgba(0, 0, 0, 0.4);
		width: 100%;
		height: 100%;
	}

	.hidden {
		display: none;
	}

	.load-wrap {
		display: flex;
		justify-content: center;
		align-items: center;
		height: 100%;
	}

	.loader {
		display: flex;
		position: relative;
		background-color: var(--border-color-accent-subdued);
		animation: shadowPulse 2s linear infinite;
		box-shadow:
			-24px 0 var(--border-color-accent-subdued),
			24px 0 var(--border-color-accent-subdued);
		margin: var(--spacing-md);
		border-radius: 50%;
		width: 10px;
		height: 10px;
		scale: 0.5;
	}

	@keyframes shadowPulse {
		33% {
			box-shadow:
				-24px 0 var(--border-color-accent-subdued),
				24px 0 #fff;
			background: #fff;
		}
		66% {
			box-shadow:
				-24px 0 #fff,
				24px 0 #fff;
			background: var(--border-color-accent-subdued);
		}
		100% {
			box-shadow:
				-24px 0 #fff,
				24px 0 var(--border-color-accent-subdued);
			background: #fff;
		}
	}
</style>
