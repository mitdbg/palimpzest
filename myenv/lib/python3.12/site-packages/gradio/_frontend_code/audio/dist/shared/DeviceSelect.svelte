<script>import RecordPlugin from "wavesurfer.js/dist/plugins/record.js";
import { createEventDispatcher } from "svelte";
export let i18n;
export let micDevices = [];
const dispatch = createEventDispatcher();
$:
  if (typeof window !== "undefined") {
    try {
      let tempDevices = [];
      RecordPlugin.getAvailableAudioDevices().then(
        (devices) => {
          micDevices = devices;
          devices.forEach((device) => {
            if (device.deviceId) {
              tempDevices.push(device);
            }
          });
          micDevices = tempDevices;
        }
      );
    } catch (err) {
      if (err instanceof DOMException && err.name == "NotAllowedError") {
        dispatch("error", i18n("audio.allow_recording_access"));
      }
      throw err;
    }
  }
</script>

<select
	class="mic-select"
	aria-label="Select input device"
	disabled={micDevices.length === 0}
>
	{#if micDevices.length === 0}
		<option value="">{i18n("audio.no_microphone")}</option>
	{:else}
		{#each micDevices as micDevice}
			<option value={micDevice.deviceId}>{micDevice.label}</option>
		{/each}
	{/if}
</select>

<style>
	.mic-select {
		height: var(--size-8);
		background: var(--block-background-fill);
		padding: 0px var(--spacing-xxl);
		border-radius: var(--button-large-radius);
		font-size: var(--text-md);
		border: 1px solid var(--block-border-color);
		gap: var(--size-1);
	}

	select {
		text-overflow: ellipsis;
		max-width: var(--size-40);
	}

	@media (max-width: 375px) {
		select {
			width: 100%;
		}
	}
</style>
