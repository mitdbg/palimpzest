<script>import { onMount } from "svelte";
import * as BABYLON from "babylonjs";
import * as BABYLON_LOADERS from "babylonjs-loaders";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
$: {
  if (BABYLON_LOADERS.OBJFileLoader != void 0 && !BABYLON_LOADERS.OBJFileLoader.IMPORT_VERTEX_COLORS) {
    BABYLON_LOADERS.OBJFileLoader.IMPORT_VERTEX_COLORS = true;
  }
}
export let value;
export let display_mode;
export let clear_color;
export let camera_position;
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
let engine;
let point_cloud_system = null;
let mounted = false;
onMount(() => {
  engine = new BABYLON.Engine(canvas, true);
  scene = new BABYLON.Scene(engine);
  scene.createDefaultCameraOrLight();
  scene.useRightHandedSystem = true;
  scene.clearColor = scene.clearColor = new BABYLON.Color4(...clear_color);
  engine.runRenderLoop(() => {
    scene.render();
  });
  function onWindowResize() {
    engine.resize();
  }
  window.addEventListener("resize", onWindowResize);
  mounted = true;
  return () => {
    scene.dispose();
    engine.dispose();
    window.removeEventListener("resize", onWindowResize);
  };
});
$:
  mounted && load_model(resolved_url);
function load_model(url2) {
  if (scene) {
    scene.meshes.forEach((mesh) => {
      mesh.dispose();
    });
    if (point_cloud_system) {
      point_cloud_system.dispose();
      point_cloud_system = null;
    }
    if (url2) {
      BABYLON.SceneLoader.ShowLoadingScreen = false;
      BABYLON.SceneLoader.Append(
        url2,
        "",
        scene,
        () => {
          if (display_mode === "point_cloud") {
            create_point_cloud(scene);
          } else if (display_mode === "wireframe") {
            create_wireframe(scene);
          } else {
            create_camera(scene, camera_position, zoom_speed, pan_speed);
          }
        },
        void 0,
        void 0,
        "." + value.path.split(".").pop()
      );
    }
  }
}
function create_camera(scene2, camera_position2, zoom_speed2, pan_speed2) {
  scene2.createDefaultCamera(true, true, true);
  var helperCamera = scene2.activeCamera;
  if (camera_position2[0] !== null) {
    helperCamera.alpha = BABYLON.Tools.ToRadians(camera_position2[0]);
  }
  if (camera_position2[1] !== null) {
    helperCamera.beta = BABYLON.Tools.ToRadians(camera_position2[1]);
  }
  if (camera_position2[2] !== null) {
    helperCamera.radius = camera_position2[2];
  }
  helperCamera.lowerRadiusLimit = 0.1;
  const updateCameraSensibility = () => {
    helperCamera.wheelPrecision = 250 / (helperCamera.radius * zoom_speed2);
    helperCamera.panningSensibility = 1e4 * pan_speed2 / helperCamera.radius;
  };
  updateCameraSensibility();
  helperCamera.attachControl(true);
  helperCamera.onAfterCheckInputsObservable.add(updateCameraSensibility);
}
export function reset_camera_position(camera_position2, zoom_speed2, pan_speed2) {
  if (scene) {
    scene.removeCamera(scene.activeCamera);
    create_camera(scene, camera_position2, zoom_speed2, pan_speed2);
  }
}
function create_point_cloud(scene2) {
  const meshes = scene2.meshes;
  const pointPositions = [];
  meshes.forEach((mesh) => {
    if (mesh instanceof BABYLON.Mesh) {
      const positions = mesh.getVerticesData(
        BABYLON.VertexBuffer.PositionKind
      );
      if (positions) {
        for (let i = 0; i < positions.length; i += 3) {
          pointPositions.push(
            new BABYLON.Vector3(
              positions[i],
              positions[i + 1],
              positions[i + 2]
            )
          );
        }
      }
      mesh.setEnabled(false);
    }
  });
  point_cloud_system = new BABYLON.PointsCloudSystem(
    "point_cloud_system",
    1,
    scene2
  );
  point_cloud_system.addPoints(
    pointPositions.length,
    (particle, i) => {
      particle.position = pointPositions[i];
      particle.color = new BABYLON.Color4(
        Math.random(),
        Math.random(),
        Math.random(),
        1
      );
    }
  );
  point_cloud_system.buildMeshAsync().then((mesh) => {
    mesh.alwaysSelectAsActiveMesh = true;
    create_camera(scene2, camera_position, zoom_speed, pan_speed);
  });
}
function create_wireframe(scene2) {
  scene2.meshes.forEach((mesh) => {
    if (mesh instanceof BABYLON.Mesh) {
      mesh.material = new BABYLON.StandardMaterial(
        "wireframeMaterial",
        scene2
      );
      mesh.material.wireframe = true;
    }
    create_camera(scene2, camera_position, zoom_speed, pan_speed);
  });
}
</script>

<canvas bind:this={canvas}></canvas>
