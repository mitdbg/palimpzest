<script>import { Block } from "@gradio/atoms";
export let elem_id;
export let elem_classes;
export let visible = true;
</script>

<Block {elem_id} {elem_classes} {visible} explicit_call>
	<slot />
</Block>
