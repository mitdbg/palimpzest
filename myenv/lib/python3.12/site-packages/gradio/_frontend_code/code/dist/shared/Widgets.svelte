<script>import Copy from "./Copy.svelte";
import Download from "./Download.svelte";
import { IconButtonWrapper } from "@gradio/atoms";
export let value;
export let language;
</script>

<IconButtonWrapper>
	<Download {value} {language} />
	<Copy {value} />
</IconButtonWrapper>
