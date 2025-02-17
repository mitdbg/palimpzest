<script>import { onDestroy } from "svelte";
export let gradio;
export let value = 1;
export let active = true;
let old_value;
let old_active;
let interval;
$:
  if (old_value !== value || active !== old_active) {
    if (interval)
      clearInterval(interval);
    if (active) {
      interval = setInterval(() => {
        if (document.visibilityState === "visible")
          gradio.dispatch("tick");
      }, value * 1e3);
    }
    old_value = value;
    old_active = active;
  }
onDestroy(() => {
  if (interval)
    clearInterval(interval);
});
</script>
