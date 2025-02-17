<script>import { createEventDispatcher, onMount } from "svelte";
import { getWorkerProxyContext } from "./context";
import { should_proxy_wasm_src } from "./file-url";
import { getHeaderValue } from "../src/http";
export let href = void 0;
export let download;
const dispatch = createEventDispatcher();
let is_downloading = false;
const worker_proxy = getWorkerProxyContext();
async function wasm_click_handler() {
  if (is_downloading) {
    return;
  }
  dispatch("click");
  if (href == null) {
    throw new Error("href is not defined.");
  }
  if (worker_proxy == null) {
    throw new Error("Wasm worker proxy is not available.");
  }
  const url = new URL(href, window.location.href);
  const path = url.pathname;
  is_downloading = true;
  worker_proxy.httpRequest({
    method: "GET",
    path,
    headers: {},
    query_string: ""
  }).then((response) => {
    if (response.status !== 200) {
      throw new Error(`Failed to get file ${path} from the Wasm worker.`);
    }
    const blob = new Blob([response.body], {
      type: getHeaderValue(response.headers, "content-type")
    });
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = download;
    link.click();
    URL.revokeObjectURL(blobUrl);
  }).finally(() => {
    is_downloading = false;
  });
}
</script>

{#if worker_proxy && should_proxy_wasm_src(href)}
	{#if is_downloading}
		<slot />
	{:else}
		<a {...$$restProps} {href} on:click|preventDefault={wasm_click_handler}>
			<slot />
		</a>
	{/if}
{:else}
	<a
		style:position="relative"
		class="download-link"
		{href}
		target={typeof window !== "undefined" && window.__is_colab__
			? "_blank"
			: null}
		rel="noopener noreferrer"
		{download}
		{...$$restProps}
		on:click={dispatch.bind(null, "click")}
	>
		<slot />
	</a>
{/if}

<style>
	.unstyled-link {
		all: unset;
		cursor: pointer;
	}
</style>
