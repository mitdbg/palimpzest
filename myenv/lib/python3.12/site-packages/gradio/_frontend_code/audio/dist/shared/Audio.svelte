<script>import { createEventDispatcher } from "svelte";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
export let src = void 0;
let resolved_src;
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
</script>

<audio
	src={resolved_src}
	{...$$restProps}
	on:play={dispatch.bind(null, "play")}
	on:pause={dispatch.bind(null, "pause")}
	on:ended={dispatch.bind(null, "ended")}
/>
