<svelte:options accessors={true} />

<script>import { beforeUpdate } from "svelte";
import { encrypt, decrypt } from "./crypto";
import { dequal } from "dequal/lite";
export let storage_key;
export let secret;
export let default_value;
export let value = default_value;
let initialized = false;
let old_value = value;
export let gradio;
function load_value() {
  const stored = localStorage.getItem(storage_key);
  if (!stored) {
    old_value = default_value;
    value = old_value;
    return;
  }
  try {
    const decrypted = decrypt(stored, secret);
    old_value = JSON.parse(decrypted);
    value = old_value;
  } catch (e) {
    console.error("Error reading from localStorage:", e);
    old_value = default_value;
    value = old_value;
  }
}
function save_value() {
  try {
    const encrypted = encrypt(JSON.stringify(value), secret);
    localStorage.setItem(storage_key, encrypted);
    old_value = value;
  } catch (e) {
    console.error("Error writing to localStorage:", e);
  }
}
$:
  value && (() => {
    if (!dequal(value, old_value)) {
      save_value();
      gradio.dispatch("change");
    }
  })();
beforeUpdate(() => {
  if (!initialized) {
    initialized = true;
    load_value();
  }
});
</script>
