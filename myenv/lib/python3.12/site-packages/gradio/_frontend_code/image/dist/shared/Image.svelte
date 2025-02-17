<script>import { resolve_wasm_src } from "@gradio/wasm/svelte";
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
</script>

<!-- svelte-ignore a11y-missing-attribute -->
<img src={resolved_src} {...$$restProps} on:load />

<style>
	img {
		object-fit: cover;
	}
</style>
