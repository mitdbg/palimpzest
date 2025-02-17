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
  }, 1e3);
}
const copy_conversation = () => {
  if (value) {
    const conversation_value = value.map((message) => {
      if (message.type === "text") {
        return `${message.role}: ${message.content}`;
      }
      return `${message.role}: ${message.content.value.url}`;
    }).join("\n\n");
    navigator.clipboard.writeText(conversation_value).catch((err) => {
      console.error("Failed to copy conversation: ", err);
    });
  }
};
async function handle_copy() {
  if ("clipboard" in navigator) {
    copy_conversation();
    copy_feedback();
  }
}
onDestroy(() => {
  if (timer)
    clearTimeout(timer);
});
</script>

<IconButton
	Icon={copied ? Check : Copy}
	on:click={handle_copy}
	label={copied ? "Copied conversation" : "Copy conversation"}
></IconButton>
