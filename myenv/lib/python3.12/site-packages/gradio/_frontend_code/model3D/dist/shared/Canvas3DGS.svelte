<script>import { onMount } from "svelte";
import * as SPLAT from "gsplat";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
export let value;
export let zoom_speed;
export let pan_speed;
$:
  url = value.url;
export let resolved_url = void 0;
let latest_url;
$: {
  resolved_url = url;
  if (url) {
    latest_url = url;
    const resolving_url = url;
    resolve_wasm_src(url).then((resolved) => {
      if (latest_url === resolving_url) {
        resolved_url = resolved ?? void 0;
      } else {
        resolved && URL.revokeObjectURL(resolved);
      }
    });
  }
}
let canvas;
let scene;
let camera;
let renderer = null;
let controls;
let mounted = false;
let frameId = null;
function reset_scene() {
  if (frameId !== null) {
    cancelAnimationFrame(frameId);
    frameId = null;
  }
  if (renderer !== null) {
    renderer.dispose();
    renderer = null;
  }
  scene = new SPLAT.Scene();
  camera = new SPLAT.Camera();
  renderer = new SPLAT.WebGLRenderer(canvas);
  controls = new SPLAT.OrbitControls(camera, canvas);
  controls.zoomSpeed = zoom_speed;
  controls.panSpeed = pan_speed;
  if (!value) {
    return;
  }
  let loading = false;
  const load = async () => {
    if (loading) {
      console.error("Already loading");
      return;
    }
    if (!resolved_url) {
      throw new Error("No resolved URL");
    }
    loading = true;
    if (resolved_url.endsWith(".ply")) {
      await SPLAT.PLYLoader.LoadAsync(resolved_url, scene, void 0);
    } else if (resolved_url.endsWith(".splat")) {
      await SPLAT.Loader.LoadAsync(resolved_url, scene, void 0);
    } else {
      throw new Error("Unsupported file type");
    }
    loading = false;
  };
  const frame = () => {
    if (!renderer) {
      return;
    }
    if (loading) {
      frameId = requestAnimationFrame(frame);
      return;
    }
    controls.update();
    renderer.render(scene, camera);
    frameId = requestAnimationFrame(frame);
  };
  load();
  frameId = requestAnimationFrame(frame);
}
onMount(() => {
  if (value != null) {
    reset_scene();
  }
  mounted = true;
  return () => {
    if (renderer) {
      renderer.dispose();
    }
  };
});
$:
  ({ path } = value || {
    path: void 0
  });
$:
  canvas && mounted && path && reset_scene();
</script>

<canvas bind:this={canvas}></canvas>
