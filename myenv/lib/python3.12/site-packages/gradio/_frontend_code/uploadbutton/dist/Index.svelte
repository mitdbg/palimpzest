<script context="module">export { default as BaseUploadButton } from "./shared/UploadButton.svelte";
</script>

<script>import UploadButton from "./shared/UploadButton.svelte";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let label;
export let value;
export let file_count;
export let file_types = [];
export let root;
export let size = "lg";
export let scale = null;
export let icon = null;
export let min_width = void 0;
export let variant = "secondary";
export let gradio;
export let interactive;
$:
  disabled = !interactive;
async function handle_event(detail, event) {
  value = detail;
  gradio.dispatch(event);
}
</script>

<UploadButton
	{elem_id}
	{elem_classes}
	{visible}
	{file_count}
	{file_types}
	{size}
	{scale}
	{icon}
	{min_width}
	{root}
	{value}
	{disabled}
	{variant}
	{label}
	max_file_size={gradio.max_file_size}
	on:click={() => gradio.dispatch("click")}
	on:change={({ detail }) => handle_event(detail, "change")}
	on:upload={({ detail }) => handle_event(detail, "upload")}
	on:error={({ detail }) => {
		gradio.dispatch("error", detail);
	}}
	upload={(...args) => gradio.client.upload(...args)}
>
	{label ? gradio.i18n(label) : ""}
</UploadButton>
