<script>import { createEventDispatcher, onMount } from "svelte";
import {
  EditorView,
  ViewUpdate,
  keymap,
  placeholder as placeholderExt
} from "@codemirror/view";
import { StateEffect, EditorState } from "@codemirror/state";
import { indentWithTab } from "@codemirror/commands";
import { basicDark } from "cm6-theme-basic-dark";
import { basicLight } from "cm6-theme-basic-light";
import { basicSetup } from "./extensions";
import { getLanguageExtension } from "./language";
export let class_names = "";
export let value = "";
export let dark_mode;
export let basic = true;
export let language;
export let lines = 5;
export let max_lines = null;
export let extensions = [];
export let use_tab = true;
export let readonly = false;
export let placeholder = void 0;
export let wrap_lines = false;
const dispatch = createEventDispatcher();
let lang_extension;
let element;
let view;
$:
  get_lang(language);
async function get_lang(val) {
  const ext = await getLanguageExtension(val);
  lang_extension = ext;
}
$:
  reconfigure(), lang_extension, readonly;
$:
  set_doc(value);
$:
  update_lines();
function set_doc(new_doc) {
  if (view && new_doc !== view.state.doc.toString()) {
    view.dispatch({
      changes: {
        from: 0,
        to: view.state.doc.length,
        insert: new_doc
      }
    });
  }
}
function update_lines() {
  if (view) {
    view.requestMeasure({ read: resize });
  }
}
function create_editor_view() {
  const editorView = new EditorView({
    parent: element,
    state: create_editor_state(value)
  });
  editorView.dom.addEventListener("focus", handle_focus, true);
  editorView.dom.addEventListener("blur", handle_blur, true);
  return editorView;
}
function handle_focus() {
  dispatch("focus");
}
function handle_blur() {
  dispatch("blur");
}
function getGutterLineHeight(_view) {
  let elements = _view.dom.querySelectorAll(".cm-gutterElement");
  if (elements.length === 0) {
    return null;
  }
  for (var i = 0; i < elements.length; i++) {
    let node = elements[i];
    let height = getComputedStyle(node)?.height ?? "0px";
    if (height != "0px") {
      return height;
    }
  }
  return null;
}
function resize(_view) {
  let scroller = _view.dom.querySelector(".cm-scroller");
  if (!scroller) {
    return null;
  }
  const lineHeight = getGutterLineHeight(_view);
  if (!lineHeight) {
    return null;
  }
  const minLines = lines == 1 ? 1 : lines + 1;
  scroller.style.minHeight = `calc(${lineHeight} * ${minLines})`;
  if (max_lines)
    scroller.style.maxHeight = `calc(${lineHeight} * ${max_lines + 1})`;
}
function handle_change(vu) {
  if (vu.docChanged) {
    const doc = vu.state.doc;
    const text = doc.toString();
    value = text;
    dispatch("change", text);
  }
  view.requestMeasure({ read: resize });
}
function get_extensions() {
  const stateExtensions = [
    ...get_base_extensions(
      basic,
      use_tab,
      placeholder,
      readonly,
      lang_extension
    ),
    FontTheme,
    ...get_theme(),
    ...extensions
  ];
  return stateExtensions;
}
const FontTheme = EditorView.theme({
  "&": {
    fontSize: "var(--text-sm)",
    backgroundColor: "var(--border-color-secondary)"
  },
  ".cm-content": {
    paddingTop: "5px",
    paddingBottom: "5px",
    color: "var(--body-text-color)",
    fontFamily: "var(--font-mono)",
    minHeight: "100%"
  },
  ".cm-gutterElement": {
    marginRight: "var(--spacing-xs)"
  },
  ".cm-gutters": {
    marginRight: "1px",
    borderRight: "1px solid var(--border-color-primary)",
    backgroundColor: "var(--block-background-fill);",
    color: "var(--body-text-color-subdued)"
  },
  ".cm-focused": {
    outline: "none"
  },
  ".cm-scroller": {
    height: "auto"
  },
  ".cm-cursor": {
    borderLeftColor: "var(--body-text-color)"
  }
});
function create_editor_state(_value) {
  return EditorState.create({
    doc: _value ?? void 0,
    extensions: get_extensions()
  });
}
function get_base_extensions(basic2, use_tab2, placeholder2, readonly2, lang) {
  const extensions2 = [
    EditorView.editable.of(!readonly2),
    EditorState.readOnly.of(readonly2),
    EditorView.contentAttributes.of({ "aria-label": "Code input container" })
  ];
  if (basic2) {
    extensions2.push(basicSetup);
  }
  if (use_tab2) {
    extensions2.push(keymap.of([indentWithTab]));
  }
  if (placeholder2) {
    extensions2.push(placeholderExt(placeholder2));
  }
  if (lang) {
    extensions2.push(lang);
  }
  extensions2.push(EditorView.updateListener.of(handle_change));
  if (wrap_lines) {
    extensions2.push(EditorView.lineWrapping);
  }
  return extensions2;
}
function get_theme() {
  const extensions2 = [];
  if (dark_mode) {
    extensions2.push(basicDark);
  } else {
    extensions2.push(basicLight);
  }
  return extensions2;
}
function reconfigure() {
  view?.dispatch({
    effects: StateEffect.reconfigure.of(get_extensions())
  });
}
onMount(() => {
  view = create_editor_view();
  return () => view?.destroy();
});
</script>

<div class="wrap">
	<div class="codemirror-wrapper {class_names}" bind:this={element} />
</div>

<style>
	.wrap {
		display: flex;
		flex-direction: column;
		flex-grow: 1;
		margin: 0;
		padding: 0;
		height: 100%;
	}
	.codemirror-wrapper {
		flex-grow: 1;
		overflow: auto;
	}

	:global(.cm-editor) {
		height: 100%;
	}

	/* Dunno why this doesn't work through the theme API -- don't remove*/
	:global(.cm-selectionBackground) {
		background-color: #b9d2ff30 !important;
	}

	:global(.cm-focused) {
		outline: none !important;
	}
</style>
