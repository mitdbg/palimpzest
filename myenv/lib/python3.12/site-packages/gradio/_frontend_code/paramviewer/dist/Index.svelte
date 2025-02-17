<script>import ParamViewer from "./ParamViewer.svelte";
export let value;
export let linkify = [];
export let header = null;
</script>

<ParamViewer docs={value} {linkify} {header} />
