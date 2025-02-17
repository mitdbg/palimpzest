<script>import { uploadToHuggingFace } from "@gradio/utils";
import { Empty } from "@gradio/atoms";
import {
  ShareButton,
  IconButton,
  BlockLabel,
  IconButtonWrapper
} from "@gradio/atoms";
import { Download, Music } from "@gradio/icons";
import AudioPlayer from "../player/AudioPlayer.svelte";
import { createEventDispatcher } from "svelte";
import { DownloadLink } from "@gradio/wasm/svelte";
export let value = null;
export let label;
export let show_label = true;
export let show_download_button = true;
export let show_share_button = false;
export let i18n;
export let waveform_settings = {};
export let waveform_options = {
  show_recording_waveform: true
};
export let editable = true;
export let loop;
export let display_icon_button_wrapper_top_corner = false;
const dispatch = createEventDispatcher();
$:
  value && dispatch("change", value);
</script>

<BlockLabel
	{show_label}
	Icon={Music}
	float={false}
	label={label || i18n("audio.audio")}
/>

{#if value !== null}
	<IconButtonWrapper
		display_top_corner={display_icon_button_wrapper_top_corner}
	>
		{#if show_download_button}
			<DownloadLink
				href={value.is_stream
					? value.url?.replace("playlist.m3u8", "playlist-file")
					: value.url}
				download={value.orig_name || value.path}
			>
				<IconButton Icon={Download} label={i18n("common.download")} />
			</DownloadLink>
		{/if}
		{#if show_share_button}
			<ShareButton
				{i18n}
				on:error
				on:share
				formatter={async (value) => {
					if (!value) return "";
					let url = await uploadToHuggingFace(value.url, "url");
					return `<audio controls src="${url}"></audio>`;
				}}
				{value}
			/>
		{/if}
	</IconButtonWrapper>

	<AudioPlayer
		{value}
		{label}
		{i18n}
		{waveform_settings}
		{waveform_options}
		{editable}
		{loop}
		on:pause
		on:play
		on:stop
		on:load
	/>
{:else}
	<Empty size="small">
		<Music />
	</Empty>
{/if}
