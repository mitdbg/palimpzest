<script>import { onDestroy } from "svelte";
import { Copy, Check } from "@gradio/icons";
import { IconButton } from "@gradio/atoms";
let copied = false;
export let value;
let timer;
function copy_feedback() {
  copied = true;
  if (timer)
    clearTimeout(timer);
  timer = setTimeout(() => {
    copied = false;
  }, 2e3);
}
async function handle_copy() {
  if ("clipboard" in navigator) {
    await navigator.clipboard.writeText(value);
    copy_feedback();
  }
}
onDestroy(() => {
  if (timer)
    clearTimeout(timer);
});
</script>

<IconButton Icon={copied ? Check : Copy} on:click={handle_copy} />
