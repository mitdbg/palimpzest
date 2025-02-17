<script>import { VolumeMuted, VolumeHigh, VolumeLow } from "@gradio/icons";
export let currentVolume;
</script>

{#if currentVolume == 0}
	<VolumeMuted />
{:else if currentVolume < 0.5}
	<VolumeLow />
{:else if currentVolume >= 0.5}
	<VolumeHigh />
{/if}
