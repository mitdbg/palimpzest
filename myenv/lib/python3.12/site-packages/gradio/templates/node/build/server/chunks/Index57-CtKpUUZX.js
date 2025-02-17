import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, e as escape, n as null_to_empty, b as add_attribute, f as each, d as add_styles, o as onDestroy, m as missing_component } from './ssr-fyTaU2Wq.js';
import { B as Block, S as Static, f as BlockLabel, C as Chat, i as IconButtonWrapper, j as IconButton, q as Community, r as Trash, t as MarkdownCode, M as Music, v as File$1, w as Copy, x as ScrollDownArrow, y as Check, z as Clear, R as Retry, A as Undo, G as Edit, H as DropdownCircularArrow } from './2-CnaXPGyd.js';
import { I as Image$1 } from './Image-BwObd70i.js';
import { d as dequal } from './index6-sfNUnwRZ.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';
import './DownloadLink-BKK_IWmU.js';

const redirect_src_url = (src, root) => src.replace('src="/file', `src="${root}file`);
function get_component_for_mime_type(mime_type) {
  if (!mime_type)
    return "file";
  if (mime_type.includes("audio"))
    return "audio";
  if (mime_type.includes("video"))
    return "video";
  if (mime_type.includes("image"))
    return "image";
  return "file";
}
function convert_file_message_to_component_message(message) {
  const _file = Array.isArray(message.file) ? message.file[0] : message.file;
  return {
    component: get_component_for_mime_type(_file?.mime_type),
    value: message.file,
    alt_text: message.alt_text,
    constructor_args: {},
    props: {}
  };
}
function normalise_messages(messages, root) {
  if (messages === null)
    return messages;
  const thought_map = /* @__PURE__ */ new Map();
  return messages.map((message, i) => {
    let normalized = typeof message.content === "string" ? {
      role: message.role,
      metadata: message.metadata,
      content: redirect_src_url(message.content, root),
      type: "text",
      index: i,
      options: message.options
    } : "file" in message.content ? {
      content: convert_file_message_to_component_message(
        message.content
      ),
      metadata: message.metadata,
      role: message.role,
      type: "component",
      index: i,
      options: message.options
    } : { type: "component", ...message };
    const { id, title, parent_id } = message.metadata || {};
    if (parent_id) {
      const parent = thought_map.get(String(parent_id));
      if (parent) {
        const thought = { ...normalized, children: [] };
        parent.children.push(thought);
        if (id && title) {
          thought_map.set(String(id), thought);
        }
        return null;
      }
    }
    if (id && title) {
      const thought = { ...normalized, children: [] };
      thought_map.set(String(id), thought);
      return thought;
    }
    return normalized;
  }).filter((msg) => msg !== null);
}
function normalise_tuples(messages, root) {
  if (messages === null)
    return messages;
  const msg = messages.flatMap((message_pair, i) => {
    return message_pair.map((message, index) => {
      if (message == null)
        return null;
      const role = index == 0 ? "user" : "assistant";
      if (typeof message === "string") {
        return {
          role,
          type: "text",
          content: redirect_src_url(message, root),
          metadata: { title: null },
          index: [i, index]
        };
      }
      if ("file" in message) {
        return {
          content: convert_file_message_to_component_message(message),
          role,
          type: "component",
          index: [i, index]
        };
      }
      return {
        role,
        content: message,
        type: "component",
        index: [i, index]
      };
    });
  });
  return msg.filter((message) => message != null);
}
function is_component_message(message) {
  return message.type === "component";
}
function is_last_bot_message(messages, all_messages) {
  const is_bot = messages[messages.length - 1].role === "assistant";
  const last_index = messages[messages.length - 1].index;
  const is_last = JSON.stringify(last_index) === JSON.stringify(all_messages[all_messages.length - 1].index);
  return is_last && is_bot;
}
function group_messages(messages, msg_format) {
  const groupedMessages = [];
  let currentGroup = [];
  let currentRole = null;
  for (const message of messages) {
    if (!(message.role === "assistant" || message.role === "user")) {
      continue;
    }
    if (message.role === currentRole) {
      currentGroup.push(message);
    } else {
      if (currentGroup.length > 0) {
        groupedMessages.push(currentGroup);
      }
      currentGroup = [message];
      currentRole = message.role;
    }
  }
  if (currentGroup.length > 0) {
    groupedMessages.push(currentGroup);
  }
  return groupedMessages;
}
async function load_components(component_names, _components, load_component) {
  let names = [];
  let components = [];
  component_names.forEach((component_name) => {
    if (_components[component_name] || component_name === "file") {
      return;
    }
    const variant = component_name === "dataframe" ? "component" : "base";
    const { name, component } = load_component(component_name, variant);
    names.push(name);
    components.push(component);
  });
  const loaded_components = await Promise.all(components);
  loaded_components.forEach((component, i) => {
    _components[names[i]] = component.default;
  });
  return _components;
}
function get_components_from_messages(messages) {
  if (!messages)
    return [];
  let components = /* @__PURE__ */ new Set();
  messages.forEach((message) => {
    if (message.type === "component") {
      components.add(message.content.component);
    }
  });
  return Array.from(components);
}
function get_thought_content(msg, depth = 0) {
  let content = "";
  const indent = "  ".repeat(depth);
  if (msg.metadata?.title) {
    content += `${indent}${depth > 0 ? "- " : ""}${msg.metadata.title}
`;
  }
  if (typeof msg.content === "string") {
    content += `${indent}  ${msg.content}
`;
  }
  const thought = msg;
  if (thought.children?.length > 0) {
    content += thought.children.map((child) => get_thought_content(child, depth + 1)).join("");
  }
  return content;
}
function all_text(message) {
  if (Array.isArray(message)) {
    return message.map((m) => {
      if (m.metadata?.title) {
        return get_thought_content(m);
      }
      return m.content;
    }).join("\n");
  }
  if (message.metadata?.title) {
    return get_thought_content(message);
  }
  return message.content;
}
function is_all_text(message) {
  return Array.isArray(message) && message.every((m) => typeof m.content === "string") || !Array.isArray(message) && typeof message.content === "string";
}
const ThumbDownActive = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11.25 6.61523H9.375V1.36523H11.25V6.61523ZM3.375 1.36523H8.625V6.91636L7.48425 8.62748L7.16737 10.8464C7.14108 11.0248 7.05166 11.1879 6.91535 11.3061C6.77904 11.4242 6.60488 11.4896 6.4245 11.4902H6.375C6.07672 11.4899 5.79075 11.3713 5.57983 11.1604C5.36892 10.9495 5.2503 10.6635 5.25 10.3652V8.11523H2.25C1.85233 8.11474 1.47109 7.95654 1.18989 7.67535C0.908691 7.39415 0.750496 7.01291 0.75 6.61523V3.99023C0.750992 3.29435 1.02787 2.62724 1.51994 2.13517C2.01201 1.64311 2.67911 1.36623 3.375 1.36523Z" fill="currentColor"></path></svg>`;
});
const ThumbDownDefault = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2.25 8.11523H4.5V10.3652C4.5003 10.6635 4.61892 10.9495 4.82983 11.1604C5.04075 11.3713 5.32672 11.4899 5.625 11.4902H6.42488C6.60519 11.4895 6.77926 11.4241 6.91549 11.3059C7.05172 11.1878 7.14109 11.0248 7.16737 10.8464L7.48425 8.62748L8.82562 6.61523H11.25V1.36523H3.375C2.67911 1.36623 2.01201 1.64311 1.51994 2.13517C1.02787 2.62724 0.750992 3.29435 0.75 3.99023V6.61523C0.750496 7.01291 0.908691 7.39415 1.18989 7.67535C1.47109 7.95654 1.85233 8.11474 2.25 8.11523ZM9 2.11523H10.5V5.86523H9V2.11523ZM1.5 3.99023C1.5006 3.49314 1.69833 3.01657 2.04983 2.66507C2.40133 2.31356 2.8779 2.11583 3.375 2.11523H8.25V6.12661L6.76575 8.35298L6.4245 10.7402H5.625C5.52554 10.7402 5.43016 10.7007 5.35983 10.6304C5.28951 10.5601 5.25 10.4647 5.25 10.3652V7.36523H2.25C2.05118 7.36494 1.86059 7.28582 1.72 7.14524C1.57941 7.00465 1.5003 6.81406 1.5 6.61523V3.99023Z" fill="currentColor"></path></svg>`;
});
const ThumbUpActive = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.75 6.24023H2.625V11.4902H0.75V6.24023ZM8.625 11.4902H3.375V5.93911L4.51575 4.22798L4.83263 2.00911C4.85892 1.83065 4.94834 1.66754 5.08465 1.5494C5.22096 1.43125 5.39512 1.36591 5.5755 1.36523H5.625C5.92328 1.36553 6.20925 1.48415 6.42017 1.69507C6.63108 1.90598 6.7497 2.19196 6.75 2.49023V4.74023H9.75C10.1477 4.74073 10.5289 4.89893 10.8101 5.18012C11.0913 5.46132 11.2495 5.84256 11.25 6.24023V8.86523C11.249 9.56112 10.9721 10.2282 10.4801 10.7203C9.98799 11.2124 9.32089 11.4892 8.625 11.4902Z" fill="currentColor"></path></svg>`;
});
const ThumbUpDefault = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9.75 4.74023H7.5V2.49023C7.4997 2.19196 7.38108 1.90598 7.17017 1.69507C6.95925 1.48415 6.67328 1.36553 6.375 1.36523H5.57512C5.39481 1.366 5.22074 1.43138 5.08451 1.54952C4.94828 1.66766 4.85891 1.83072 4.83262 2.00911L4.51575 4.22798L3.17438 6.24023H0.75V11.4902H8.625C9.32089 11.4892 9.98799 11.2124 10.4801 10.7203C10.9721 10.2282 11.249 9.56112 11.25 8.86523V6.24023C11.2495 5.84256 11.0913 5.46132 10.8101 5.18012C10.5289 4.89893 10.1477 4.74073 9.75 4.74023ZM3 10.7402H1.5V6.99023H3V10.7402ZM10.5 8.86523C10.4994 9.36233 10.3017 9.8389 9.95017 10.1904C9.59867 10.5419 9.1221 10.7396 8.625 10.7402H3.75V6.72886L5.23425 4.50248L5.5755 2.11523H6.375C6.47446 2.11523 6.56984 2.15474 6.64017 2.22507C6.71049 2.2954 6.75 2.39078 6.75 2.49023V5.49023H9.75C9.94882 5.49053 10.1394 5.56965 10.28 5.71023C10.4206 5.85082 10.4997 6.04141 10.5 6.24023V8.86523Z" fill="currentColor"></path></svg>`;
});
const Flag = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg id="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none"><path fill="currentColor" d="M6,30H4V2H28l-5.8,9L28,20H6ZM6,18H24.33L19.8,11l4.53-7H6Z"></path></svg>`;
});
const FlagActive = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg id="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none"><path fill="currentColor" d="M4,2H28l-5.8,9L28,20H6v10H4V2z"></path></svg>`;
});
const css$8 = {
  code: ".extra-feedback.svelte-14rmxes.svelte-14rmxes{display:flex;align-items:center;position:relative}.extra-feedback-options.svelte-14rmxes.svelte-14rmxes{display:none;position:absolute;padding:var(--spacing-md) 0;flex-direction:column;gap:var(--spacing-sm);top:100%}.extra-feedback.svelte-14rmxes:hover .extra-feedback-options.svelte-14rmxes{display:flex}.extra-feedback-option.svelte-14rmxes.svelte-14rmxes{border:1px solid var(--border-color-primary);border-radius:var(--radius-sm);color:var(--block-label-text-color);background-color:var(--block-background-fill);font-size:var(--text-xs);padding:var(--spacing-xxs) var(--spacing-sm);width:max-content}",
  map: '{"version":3,"file":"LikeDislike.svelte","sources":["LikeDislike.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { IconButton } from \\"@gradio/atoms\\";\\nimport ThumbDownActive from \\"./ThumbDownActive.svelte\\";\\nimport ThumbDownDefault from \\"./ThumbDownDefault.svelte\\";\\nimport ThumbUpActive from \\"./ThumbUpActive.svelte\\";\\nimport ThumbUpDefault from \\"./ThumbUpDefault.svelte\\";\\nimport Flag from \\"./Flag.svelte\\";\\nimport FlagActive from \\"./FlagActive.svelte\\";\\nexport let handle_action;\\nexport let feedback_options;\\nexport let selected = null;\\n$: extra_feedback = feedback_options.filter((option) => option !== \\"Like\\" && option !== \\"Dislike\\");\\nfunction toggleSelection(newSelection) {\\n    selected = selected === newSelection ? null : newSelection;\\n    handle_action(selected);\\n}\\n<\/script>\\n\\n{#if feedback_options.includes(\\"Like\\") || feedback_options.includes(\\"Dislike\\")}\\n\\t{#if feedback_options.includes(\\"Dislike\\")}\\n\\t\\t<IconButton\\n\\t\\t\\tIcon={selected === \\"Dislike\\" ? ThumbDownActive : ThumbDownDefault}\\n\\t\\t\\tlabel={selected === \\"Dislike\\" ? \\"clicked dislike\\" : \\"dislike\\"}\\n\\t\\t\\tcolor={selected === \\"Dislike\\"\\n\\t\\t\\t\\t? \\"var(--color-accent)\\"\\n\\t\\t\\t\\t: \\"var(--block-label-text-color)\\"}\\n\\t\\t\\ton:click={() => toggleSelection(\\"Dislike\\")}\\n\\t\\t/>\\n\\t{/if}\\n\\t{#if feedback_options.includes(\\"Like\\")}\\n\\t\\t<IconButton\\n\\t\\t\\tIcon={selected === \\"Like\\" ? ThumbUpActive : ThumbUpDefault}\\n\\t\\t\\tlabel={selected === \\"Like\\" ? \\"clicked like\\" : \\"like\\"}\\n\\t\\t\\tcolor={selected === \\"Like\\"\\n\\t\\t\\t\\t? \\"var(--color-accent)\\"\\n\\t\\t\\t\\t: \\"var(--block-label-text-color)\\"}\\n\\t\\t\\ton:click={() => toggleSelection(\\"Like\\")}\\n\\t\\t/>\\n\\t{/if}\\n{/if}\\n\\n{#if extra_feedback.length > 0}\\n\\t<div class=\\"extra-feedback no-border\\">\\n\\t\\t<IconButton\\n\\t\\t\\tIcon={selected && extra_feedback.includes(selected) ? FlagActive : Flag}\\n\\t\\t\\tlabel=\\"Feedback\\"\\n\\t\\t\\tcolor={selected && extra_feedback.includes(selected)\\n\\t\\t\\t\\t? \\"var(--color-accent)\\"\\n\\t\\t\\t\\t: \\"var(--block-label-text-color)\\"}\\n\\t\\t/>\\n\\t\\t<div class=\\"extra-feedback-options\\">\\n\\t\\t\\t{#each extra_feedback as option}\\n\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\tclass=\\"extra-feedback-option\\"\\n\\t\\t\\t\\t\\tstyle:font-weight={selected === option ? \\"bold\\" : \\"normal\\"}\\n\\t\\t\\t\\t\\ton:click={() => {\\n\\t\\t\\t\\t\\t\\ttoggleSelection(option);\\n\\t\\t\\t\\t\\t\\thandle_action(selected ? selected : null);\\n\\t\\t\\t\\t\\t}}>{option}</button\\n\\t\\t\\t\\t>\\n\\t\\t\\t{/each}\\n\\t\\t</div>\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\t.extra-feedback {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tposition: relative;\\n\\t}\\n\\t.extra-feedback-options {\\n\\t\\tdisplay: none;\\n\\t\\tposition: absolute;\\n\\t\\tpadding: var(--spacing-md) 0;\\n\\t\\tflex-direction: column;\\n\\t\\tgap: var(--spacing-sm);\\n\\t\\ttop: 100%;\\n\\t}\\n\\t.extra-feedback:hover .extra-feedback-options {\\n\\t\\tdisplay: flex;\\n\\t}\\n\\t.extra-feedback-option {\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tcolor: var(--block-label-text-color);\\n\\t\\tbackground-color: var(--block-background-fill);\\n\\t\\tfont-size: var(--text-xs);\\n\\t\\tpadding: var(--spacing-xxs) var(--spacing-sm);\\n\\t\\twidth: max-content;\\n\\t}</style>\\n"],"names":[],"mappings":"AAiEC,6CAAgB,CACf,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,QAAQ,CAAE,QACX,CACA,qDAAwB,CACvB,OAAO,CAAE,IAAI,CACb,QAAQ,CAAE,QAAQ,CAClB,OAAO,CAAE,IAAI,YAAY,CAAC,CAAC,CAAC,CAC5B,cAAc,CAAE,MAAM,CACtB,GAAG,CAAE,IAAI,YAAY,CAAC,CACtB,GAAG,CAAE,IACN,CACA,8BAAe,MAAM,CAAC,sCAAwB,CAC7C,OAAO,CAAE,IACV,CACA,oDAAuB,CACtB,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CAAC,CAC7C,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,KAAK,CAAE,IAAI,wBAAwB,CAAC,CACpC,gBAAgB,CAAE,IAAI,uBAAuB,CAAC,CAC9C,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,OAAO,CAAE,IAAI,aAAa,CAAC,CAAC,IAAI,YAAY,CAAC,CAC7C,KAAK,CAAE,WACR"}'
};
const LikeDislike = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let extra_feedback;
  let { handle_action } = $$props;
  let { feedback_options } = $$props;
  let { selected = null } = $$props;
  if ($$props.handle_action === void 0 && $$bindings.handle_action && handle_action !== void 0)
    $$bindings.handle_action(handle_action);
  if ($$props.feedback_options === void 0 && $$bindings.feedback_options && feedback_options !== void 0)
    $$bindings.feedback_options(feedback_options);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  $$result.css.add(css$8);
  extra_feedback = feedback_options.filter((option) => option !== "Like" && option !== "Dislike");
  return `${feedback_options.includes("Like") || feedback_options.includes("Dislike") ? `${feedback_options.includes("Dislike") ? `${validate_component(IconButton, "IconButton").$$render(
    $$result,
    {
      Icon: selected === "Dislike" ? ThumbDownActive : ThumbDownDefault,
      label: selected === "Dislike" ? "clicked dislike" : "dislike",
      color: selected === "Dislike" ? "var(--color-accent)" : "var(--block-label-text-color)"
    },
    {},
    {}
  )}` : ``} ${feedback_options.includes("Like") ? `${validate_component(IconButton, "IconButton").$$render(
    $$result,
    {
      Icon: selected === "Like" ? ThumbUpActive : ThumbUpDefault,
      label: selected === "Like" ? "clicked like" : "like",
      color: selected === "Like" ? "var(--color-accent)" : "var(--block-label-text-color)"
    },
    {},
    {}
  )}` : ``}` : ``} ${extra_feedback.length > 0 ? `<div class="extra-feedback no-border svelte-14rmxes">${validate_component(IconButton, "IconButton").$$render(
    $$result,
    {
      Icon: selected && extra_feedback.includes(selected) ? FlagActive : Flag,
      label: "Feedback",
      color: selected && extra_feedback.includes(selected) ? "var(--color-accent)" : "var(--block-label-text-color)"
    },
    {},
    {}
  )} <div class="extra-feedback-options svelte-14rmxes">${each(extra_feedback, (option) => {
    return `<button class="extra-feedback-option svelte-14rmxes"${add_styles({
      "font-weight": selected === option ? "bold" : "normal"
    })}>${escape(option)}</button>`;
  })}</div></div>` : ``}`;
});
const Copy_1 = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  createEventDispatcher();
  let { value } = $$props;
  onDestroy(() => {
  });
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  return `${validate_component(IconButton, "IconButton").$$render(
    $$result,
    {
      label: "Copy message",
      Icon: Copy
    },
    {},
    {}
  )}`;
});
const css$7 = {
  code: ".bubble.svelte-j7nkv7 .icon-button-wrapper{margin:0px calc(var(--spacing-xl) * 2)}.message-buttons.svelte-j7nkv7{z-index:var(--layer-1)}.message-buttons-left.svelte-j7nkv7{align-self:flex-start}.bubble.message-buttons-right.svelte-j7nkv7{align-self:flex-end}.message-buttons-right.svelte-j7nkv7 .icon-button-wrapper{margin-left:auto}.bubble.with-avatar.svelte-j7nkv7{margin-left:calc(var(--spacing-xl) * 5);margin-right:calc(var(--spacing-xl) * 5)}.panel.svelte-j7nkv7{display:flex;align-self:flex-start;z-index:var(--layer-1)}",
  map: `{"version":3,"file":"ButtonPanel.svelte","sources":["ButtonPanel.svelte"],"sourcesContent":["<script lang=\\"ts\\">import LikeDislike from \\"./LikeDislike.svelte\\";\\nimport Copy from \\"./Copy.svelte\\";\\nimport { Retry, Undo, Edit, Check, Clear } from \\"@gradio/icons\\";\\nimport { IconButtonWrapper, IconButton } from \\"@gradio/atoms\\";\\nimport { all_text, is_all_text } from \\"./utils\\";\\nexport let likeable;\\nexport let feedback_options;\\nexport let show_retry;\\nexport let show_undo;\\nexport let show_edit;\\nexport let in_edit_mode;\\nexport let show_copy_button;\\nexport let message;\\nexport let position;\\nexport let avatar;\\nexport let generating;\\nexport let current_feedback;\\nexport let handle_action;\\nexport let layout;\\nexport let dispatch;\\n$: message_text = is_all_text(message) ? all_text(message) : \\"\\";\\n$: show_copy = show_copy_button && message && is_all_text(message);\\n<\/script>\\n\\n{#if show_copy || show_retry || show_undo || show_edit || likeable}\\n\\t<div\\n\\t\\tclass=\\"message-buttons-{position} {layout} message-buttons {avatar !==\\n\\t\\t\\tnull && 'with-avatar'}\\"\\n\\t>\\n\\t\\t<IconButtonWrapper top_panel={false}>\\n\\t\\t\\t{#if in_edit_mode}\\n\\t\\t\\t\\t<IconButton\\n\\t\\t\\t\\t\\tlabel=\\"Submit\\"\\n\\t\\t\\t\\t\\tIcon={Check}\\n\\t\\t\\t\\t\\ton:click={() => handle_action(\\"edit_submit\\")}\\n\\t\\t\\t\\t\\tdisabled={generating}\\n\\t\\t\\t\\t/>\\n\\t\\t\\t\\t<IconButton\\n\\t\\t\\t\\t\\tlabel=\\"Cancel\\"\\n\\t\\t\\t\\t\\tIcon={Clear}\\n\\t\\t\\t\\t\\ton:click={() => handle_action(\\"edit_cancel\\")}\\n\\t\\t\\t\\t\\tdisabled={generating}\\n\\t\\t\\t\\t/>\\n\\t\\t\\t{:else}\\n\\t\\t\\t\\t{#if show_copy}\\n\\t\\t\\t\\t\\t<Copy\\n\\t\\t\\t\\t\\t\\tvalue={message_text}\\n\\t\\t\\t\\t\\t\\ton:copy={(e) => dispatch(\\"copy\\", e.detail)}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t{#if show_retry}\\n\\t\\t\\t\\t\\t<IconButton\\n\\t\\t\\t\\t\\t\\tIcon={Retry}\\n\\t\\t\\t\\t\\t\\tlabel=\\"Retry\\"\\n\\t\\t\\t\\t\\t\\ton:click={() => handle_action(\\"retry\\")}\\n\\t\\t\\t\\t\\t\\tdisabled={generating}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t{#if show_undo}\\n\\t\\t\\t\\t\\t<IconButton\\n\\t\\t\\t\\t\\t\\tlabel=\\"Undo\\"\\n\\t\\t\\t\\t\\t\\tIcon={Undo}\\n\\t\\t\\t\\t\\t\\ton:click={() => handle_action(\\"undo\\")}\\n\\t\\t\\t\\t\\t\\tdisabled={generating}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t{#if show_edit}\\n\\t\\t\\t\\t\\t<IconButton\\n\\t\\t\\t\\t\\t\\tlabel=\\"Edit\\"\\n\\t\\t\\t\\t\\t\\tIcon={Edit}\\n\\t\\t\\t\\t\\t\\ton:click={() => handle_action(\\"edit\\")}\\n\\t\\t\\t\\t\\t\\tdisabled={generating}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t{#if likeable}\\n\\t\\t\\t\\t\\t<LikeDislike\\n\\t\\t\\t\\t\\t\\t{handle_action}\\n\\t\\t\\t\\t\\t\\t{feedback_options}\\n\\t\\t\\t\\t\\t\\tselected={current_feedback}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t{/if}\\n\\t\\t</IconButtonWrapper>\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\t.bubble :global(.icon-button-wrapper) {\\n\\t\\tmargin: 0px calc(var(--spacing-xl) * 2);\\n\\t}\\n\\n\\t.message-buttons {\\n\\t\\tz-index: var(--layer-1);\\n\\t}\\n\\t.message-buttons-left {\\n\\t\\talign-self: flex-start;\\n\\t}\\n\\n\\t.bubble.message-buttons-right {\\n\\t\\talign-self: flex-end;\\n\\t}\\n\\n\\t.message-buttons-right :global(.icon-button-wrapper) {\\n\\t\\tmargin-left: auto;\\n\\t}\\n\\n\\t.bubble.with-avatar {\\n\\t\\tmargin-left: calc(var(--spacing-xl) * 5);\\n\\t\\tmargin-right: calc(var(--spacing-xl) * 5);\\n\\t}\\n\\n\\t.panel {\\n\\t\\tdisplay: flex;\\n\\t\\talign-self: flex-start;\\n\\t\\tz-index: var(--layer-1);\\n\\t}</style>\\n"],"names":[],"mappings":"AAuFC,qBAAO,CAAS,oBAAsB,CACrC,MAAM,CAAE,GAAG,CAAC,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CACvC,CAEA,8BAAiB,CAChB,OAAO,CAAE,IAAI,SAAS,CACvB,CACA,mCAAsB,CACrB,UAAU,CAAE,UACb,CAEA,OAAO,oCAAuB,CAC7B,UAAU,CAAE,QACb,CAEA,oCAAsB,CAAS,oBAAsB,CACpD,WAAW,CAAE,IACd,CAEA,OAAO,0BAAa,CACnB,WAAW,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,CACxC,YAAY,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CACzC,CAEA,oBAAO,CACN,OAAO,CAAE,IAAI,CACb,UAAU,CAAE,UAAU,CACtB,OAAO,CAAE,IAAI,SAAS,CACvB"}`
};
const ButtonPanel = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let message_text;
  let show_copy;
  let { likeable } = $$props;
  let { feedback_options } = $$props;
  let { show_retry } = $$props;
  let { show_undo } = $$props;
  let { show_edit } = $$props;
  let { in_edit_mode } = $$props;
  let { show_copy_button } = $$props;
  let { message } = $$props;
  let { position } = $$props;
  let { avatar } = $$props;
  let { generating } = $$props;
  let { current_feedback } = $$props;
  let { handle_action } = $$props;
  let { layout } = $$props;
  let { dispatch } = $$props;
  if ($$props.likeable === void 0 && $$bindings.likeable && likeable !== void 0)
    $$bindings.likeable(likeable);
  if ($$props.feedback_options === void 0 && $$bindings.feedback_options && feedback_options !== void 0)
    $$bindings.feedback_options(feedback_options);
  if ($$props.show_retry === void 0 && $$bindings.show_retry && show_retry !== void 0)
    $$bindings.show_retry(show_retry);
  if ($$props.show_undo === void 0 && $$bindings.show_undo && show_undo !== void 0)
    $$bindings.show_undo(show_undo);
  if ($$props.show_edit === void 0 && $$bindings.show_edit && show_edit !== void 0)
    $$bindings.show_edit(show_edit);
  if ($$props.in_edit_mode === void 0 && $$bindings.in_edit_mode && in_edit_mode !== void 0)
    $$bindings.in_edit_mode(in_edit_mode);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.message === void 0 && $$bindings.message && message !== void 0)
    $$bindings.message(message);
  if ($$props.position === void 0 && $$bindings.position && position !== void 0)
    $$bindings.position(position);
  if ($$props.avatar === void 0 && $$bindings.avatar && avatar !== void 0)
    $$bindings.avatar(avatar);
  if ($$props.generating === void 0 && $$bindings.generating && generating !== void 0)
    $$bindings.generating(generating);
  if ($$props.current_feedback === void 0 && $$bindings.current_feedback && current_feedback !== void 0)
    $$bindings.current_feedback(current_feedback);
  if ($$props.handle_action === void 0 && $$bindings.handle_action && handle_action !== void 0)
    $$bindings.handle_action(handle_action);
  if ($$props.layout === void 0 && $$bindings.layout && layout !== void 0)
    $$bindings.layout(layout);
  if ($$props.dispatch === void 0 && $$bindings.dispatch && dispatch !== void 0)
    $$bindings.dispatch(dispatch);
  $$result.css.add(css$7);
  message_text = is_all_text(message) ? all_text(message) : "";
  show_copy = show_copy_button && message && is_all_text(message);
  return `${show_copy || show_retry || show_undo || show_edit || likeable ? `<div class="${"message-buttons-" + escape(position, true) + " " + escape(layout, true) + " message-buttons " + escape(avatar !== null && "with-avatar", true) + " svelte-j7nkv7"}">${validate_component(IconButtonWrapper, "IconButtonWrapper").$$render($$result, { top_panel: false }, {}, {
    default: () => {
      return `${in_edit_mode ? `${validate_component(IconButton, "IconButton").$$render(
        $$result,
        {
          label: "Submit",
          Icon: Check,
          disabled: generating
        },
        {},
        {}
      )} ${validate_component(IconButton, "IconButton").$$render(
        $$result,
        {
          label: "Cancel",
          Icon: Clear,
          disabled: generating
        },
        {},
        {}
      )}` : `${show_copy ? `${validate_component(Copy_1, "Copy").$$render($$result, { value: message_text }, {}, {})}` : ``} ${show_retry ? `${validate_component(IconButton, "IconButton").$$render(
        $$result,
        {
          Icon: Retry,
          label: "Retry",
          disabled: generating
        },
        {},
        {}
      )}` : ``} ${show_undo ? `${validate_component(IconButton, "IconButton").$$render(
        $$result,
        {
          label: "Undo",
          Icon: Undo,
          disabled: generating
        },
        {},
        {}
      )}` : ``} ${show_edit ? `${validate_component(IconButton, "IconButton").$$render(
        $$result,
        {
          label: "Edit",
          Icon: Edit,
          disabled: generating
        },
        {},
        {}
      )}` : ``} ${likeable ? `${validate_component(LikeDislike, "LikeDislike").$$render(
        $$result,
        {
          handle_action,
          feedback_options,
          selected: current_feedback
        },
        {},
        {}
      )}` : ``}`}`;
    }
  })}</div>` : ``}`;
});
const Component = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { type } = $$props;
  let { components } = $$props;
  let { value } = $$props;
  let { target } = $$props;
  let { theme_mode } = $$props;
  let { props } = $$props;
  let { i18n } = $$props;
  let { upload } = $$props;
  let { _fetch } = $$props;
  let { allow_file_downloads } = $$props;
  let { display_icon_button_wrapper_top_corner = false } = $$props;
  if ($$props.type === void 0 && $$bindings.type && type !== void 0)
    $$bindings.type(type);
  if ($$props.components === void 0 && $$bindings.components && components !== void 0)
    $$bindings.components(components);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.target === void 0 && $$bindings.target && target !== void 0)
    $$bindings.target(target);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props.props === void 0 && $$bindings.props && props !== void 0)
    $$bindings.props(props);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props._fetch === void 0 && $$bindings._fetch && _fetch !== void 0)
    $$bindings._fetch(_fetch);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  if ($$props.display_icon_button_wrapper_top_corner === void 0 && $$bindings.display_icon_button_wrapper_top_corner && display_icon_button_wrapper_top_corner !== void 0)
    $$bindings.display_icon_button_wrapper_top_corner(display_icon_button_wrapper_top_corner);
  return `${type === "gallery" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      display_icon_button_wrapper_top_corner,
      show_label: false,
      i18n,
      label: "",
      _fetch,
      allow_preview: false,
      interactive: false,
      mode: "minimal",
      fixed_height: 1
    },
    {},
    {}
  )}` : `${type === "dataframe" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      show_label: false,
      i18n,
      label: "",
      interactive: false,
      line_breaks: props.line_breaks,
      wrap: true,
      root: "",
      gradio: {
        dispatch: () => {
        }
      },
      datatype: props.datatype,
      latex_delimiters: props.latex_delimiters,
      col_count: props.col_count,
      row_count: props.row_count
    },
    {},
    {}
  )}` : `${type === "plot" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      target,
      theme_mode,
      bokeh_version: props.bokeh_version,
      caption: "",
      show_actions_button: true
    },
    {},
    {}
  )}` : `${type === "audio" ? `<div style="position: relative;">${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      show_label: false,
      show_share_button: true,
      i18n,
      label: "",
      waveform_settings: { autoplay: props.autoplay },
      show_download_button: allow_file_downloads,
      display_icon_button_wrapper_top_corner
    },
    {},
    {}
  )}</div>` : `${type === "video" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      autoplay: props.autoplay,
      value: value.video || value,
      show_label: false,
      show_share_button: true,
      i18n,
      upload,
      display_icon_button_wrapper_top_corner,
      show_download_button: allow_file_downloads
    },
    {},
    {
      default: () => {
        return `<track kind="captions">`;
      }
    }
  )}` : `${type === "image" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      show_label: false,
      label: "chatbot-image",
      show_download_button: allow_file_downloads,
      display_icon_button_wrapper_top_corner,
      i18n
    },
    {},
    {}
  )}` : `${type === "html" ? `${validate_component(components[type] || missing_component, "svelte:component").$$render(
    $$result,
    {
      value,
      show_label: false,
      label: "chatbot-image",
      show_share_button: true,
      i18n,
      gradio: {
        dispatch: () => {
        }
      }
    },
    {},
    {}
  )}` : ``}`}`}`}`}`}`}`;
});
const css$6 = {
  code: ".file-container.svelte-ulpe0d{display:flex;align-items:center;gap:var(--spacing-lg);padding:var(--spacing-lg);border-radius:var(--radius-lg);width:fit-content;margin:var(--spacing-sm) 0}.file-icon.svelte-ulpe0d{display:flex;align-items:center;justify-content:center;color:var(--body-text-color)}.file-icon.svelte-ulpe0d svg{width:var(--size-7);height:var(--size-7)}.file-info.svelte-ulpe0d{display:flex;flex-direction:column}.file-link.svelte-ulpe0d{text-decoration:none;color:var(--body-text-color);display:flex;flex-direction:column;gap:var(--spacing-xs)}.file-name.svelte-ulpe0d{font-family:var(--font);font-size:var(--text-md);font-weight:500}.file-type.svelte-ulpe0d{font-family:var(--font);font-size:var(--text-sm);color:var(--body-text-color-subdued);text-transform:uppercase}",
  map: '{"version":3,"file":"MessageContent.svelte","sources":["MessageContent.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { File } from \\"@gradio/icons\\";\\nimport Component from \\"./Component.svelte\\";\\nimport { MarkdownCode as Markdown } from \\"@gradio/markdown-code\\";\\nexport let latex_delimiters;\\nexport let sanitize_html;\\nexport let _fetch;\\nexport let i18n;\\nexport let line_breaks;\\nexport let upload;\\nexport let target;\\nexport let root;\\nexport let theme_mode;\\nexport let _components;\\nexport let render_markdown;\\nexport let scroll;\\nexport let allow_file_downloads;\\nexport let display_consecutive_in_same_bubble;\\nexport let thought_index;\\nexport let message;\\n<\/script>\\n\\n{#if message.type === \\"text\\"}\\n\\t<div class=\\"message-content\\">\\n\\t\\t<Markdown\\n\\t\\t\\tmessage={message.content}\\n\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t{sanitize_html}\\n\\t\\t\\t{render_markdown}\\n\\t\\t\\t{line_breaks}\\n\\t\\t\\ton:load={scroll}\\n\\t\\t\\t{root}\\n\\t\\t/>\\n\\t</div>\\n{:else if message.type === \\"component\\" && message.content.component in _components}\\n\\t<Component\\n\\t\\t{target}\\n\\t\\t{theme_mode}\\n\\t\\tprops={message.content.props}\\n\\t\\ttype={message.content.component}\\n\\t\\tcomponents={_components}\\n\\t\\tvalue={message.content.value}\\n\\t\\tdisplay_icon_button_wrapper_top_corner={thought_index > 0 &&\\n\\t\\t\\tdisplay_consecutive_in_same_bubble}\\n\\t\\t{i18n}\\n\\t\\t{upload}\\n\\t\\t{_fetch}\\n\\t\\ton:load={() => scroll()}\\n\\t\\t{allow_file_downloads}\\n\\t/>\\n{:else if message.type === \\"component\\" && message.content.component === \\"file\\"}\\n\\t<div class=\\"file-container\\">\\n\\t\\t<div class=\\"file-icon\\">\\n\\t\\t\\t<File />\\n\\t\\t</div>\\n\\t\\t<div class=\\"file-info\\">\\n\\t\\t\\t<a\\n\\t\\t\\t\\tdata-testid=\\"chatbot-file\\"\\n\\t\\t\\t\\tclass=\\"file-link\\"\\n\\t\\t\\t\\thref={message.content.value.url}\\n\\t\\t\\t\\ttarget=\\"_blank\\"\\n\\t\\t\\t\\tdownload={window.__is_colab__\\n\\t\\t\\t\\t\\t? null\\n\\t\\t\\t\\t\\t: message.content.value?.orig_name ||\\n\\t\\t\\t\\t\\t\\tmessage.content.value?.path.split(\\"/\\").pop() ||\\n\\t\\t\\t\\t\\t\\t\\"file\\"}\\n\\t\\t\\t>\\n\\t\\t\\t\\t<span class=\\"file-name\\"\\n\\t\\t\\t\\t\\t>{message.content.value?.orig_name ||\\n\\t\\t\\t\\t\\t\\tmessage.content.value?.path.split(\\"/\\").pop() ||\\n\\t\\t\\t\\t\\t\\t\\"file\\"}</span\\n\\t\\t\\t\\t>\\n\\t\\t\\t</a>\\n\\t\\t\\t<span class=\\"file-type\\"\\n\\t\\t\\t\\t>{(\\n\\t\\t\\t\\t\\tmessage.content.value?.orig_name ||\\n\\t\\t\\t\\t\\tmessage.content.value?.path ||\\n\\t\\t\\t\\t\\t\\"\\"\\n\\t\\t\\t\\t)\\n\\t\\t\\t\\t\\t.split(\\".\\")\\n\\t\\t\\t\\t\\t.pop()\\n\\t\\t\\t\\t\\t.toUpperCase()}</span\\n\\t\\t\\t>\\n\\t\\t</div>\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\t.file-container {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tgap: var(--spacing-lg);\\n\\t\\tpadding: var(--spacing-lg);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\twidth: fit-content;\\n\\t\\tmargin: var(--spacing-sm) 0;\\n\\t}\\n\\n\\t.file-icon {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.file-icon :global(svg) {\\n\\t\\twidth: var(--size-7);\\n\\t\\theight: var(--size-7);\\n\\t}\\n\\n\\t.file-info {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t}\\n\\n\\t.file-link {\\n\\t\\ttext-decoration: none;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tgap: var(--spacing-xs);\\n\\t}\\n\\n\\t.file-name {\\n\\t\\tfont-family: var(--font);\\n\\t\\tfont-size: var(--text-md);\\n\\t\\tfont-weight: 500;\\n\\t}\\n\\n\\t.file-type {\\n\\t\\tfont-family: var(--font);\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t\\ttext-transform: uppercase;\\n\\t}</style>\\n"],"names":[],"mappings":"AAuFC,6BAAgB,CACf,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,GAAG,CAAE,IAAI,YAAY,CAAC,CACtB,OAAO,CAAE,IAAI,YAAY,CAAC,CAC1B,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,KAAK,CAAE,WAAW,CAClB,MAAM,CAAE,IAAI,YAAY,CAAC,CAAC,CAC3B,CAEA,wBAAW,CACV,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAAM,CACvB,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,wBAAU,CAAS,GAAK,CACvB,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CACrB,CAEA,wBAAW,CACV,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MACjB,CAEA,wBAAW,CACV,eAAe,CAAE,IAAI,CACrB,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MAAM,CACtB,GAAG,CAAE,IAAI,YAAY,CACtB,CAEA,wBAAW,CACV,WAAW,CAAE,IAAI,MAAM,CAAC,CACxB,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,WAAW,CAAE,GACd,CAEA,wBAAW,CACV,WAAW,CAAE,IAAI,MAAM,CAAC,CACxB,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,KAAK,CAAE,IAAI,yBAAyB,CAAC,CACrC,cAAc,CAAE,SACjB"}'
};
const MessageContent = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { latex_delimiters } = $$props;
  let { sanitize_html } = $$props;
  let { _fetch } = $$props;
  let { i18n } = $$props;
  let { line_breaks } = $$props;
  let { upload } = $$props;
  let { target } = $$props;
  let { root } = $$props;
  let { theme_mode } = $$props;
  let { _components } = $$props;
  let { render_markdown } = $$props;
  let { scroll: scroll2 } = $$props;
  let { allow_file_downloads } = $$props;
  let { display_consecutive_in_same_bubble } = $$props;
  let { thought_index } = $$props;
  let { message } = $$props;
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props._fetch === void 0 && $$bindings._fetch && _fetch !== void 0)
    $$bindings._fetch(_fetch);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.target === void 0 && $$bindings.target && target !== void 0)
    $$bindings.target(target);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props._components === void 0 && $$bindings._components && _components !== void 0)
    $$bindings._components(_components);
  if ($$props.render_markdown === void 0 && $$bindings.render_markdown && render_markdown !== void 0)
    $$bindings.render_markdown(render_markdown);
  if ($$props.scroll === void 0 && $$bindings.scroll && scroll2 !== void 0)
    $$bindings.scroll(scroll2);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  if ($$props.display_consecutive_in_same_bubble === void 0 && $$bindings.display_consecutive_in_same_bubble && display_consecutive_in_same_bubble !== void 0)
    $$bindings.display_consecutive_in_same_bubble(display_consecutive_in_same_bubble);
  if ($$props.thought_index === void 0 && $$bindings.thought_index && thought_index !== void 0)
    $$bindings.thought_index(thought_index);
  if ($$props.message === void 0 && $$bindings.message && message !== void 0)
    $$bindings.message(message);
  $$result.css.add(css$6);
  return `${message.type === "text" ? `<div class="message-content">${validate_component(MarkdownCode, "Markdown").$$render(
    $$result,
    {
      message: message.content,
      latex_delimiters,
      sanitize_html,
      render_markdown,
      line_breaks,
      root
    },
    {},
    {}
  )}</div>` : `${message.type === "component" && message.content.component in _components ? `${validate_component(Component, "Component").$$render(
    $$result,
    {
      target,
      theme_mode,
      props: message.content.props,
      type: message.content.component,
      components: _components,
      value: message.content.value,
      display_icon_button_wrapper_top_corner: thought_index > 0 && display_consecutive_in_same_bubble,
      i18n,
      upload,
      _fetch,
      allow_file_downloads
    },
    {},
    {}
  )}` : `${message.type === "component" && message.content.component === "file" ? `<div class="file-container svelte-ulpe0d"><div class="file-icon svelte-ulpe0d">${validate_component(File$1, "File").$$render($$result, {}, {}, {})}</div> <div class="file-info svelte-ulpe0d"><a data-testid="chatbot-file" class="file-link svelte-ulpe0d"${add_attribute("href", message.content.value.url, 0)} target="_blank"${add_attribute(
    "download",
    window.__is_colab__ ? null : message.content.value?.orig_name || message.content.value?.path.split("/").pop() || "file",
    0
  )}><span class="file-name svelte-ulpe0d">${escape(message.content.value?.orig_name || message.content.value?.path.split("/").pop() || "file")}</span></a> <span class="file-type svelte-ulpe0d">${escape((message.content.value?.orig_name || message.content.value?.path || "").split(".").pop().toUpperCase())}</span></div></div>` : ``}`}`}`;
});
const css$5 = {
  code: ".thought-group.svelte-i6yrue{background:var(--background-fill-primary);border:1px solid var(--border-color-primary);border-radius:var(--radius-sm);padding:var(--spacing-md);margin:var(--spacing-md) 0;font-size:var(--text-sm)}.children.svelte-i6yrue .thought-group{border:none;margin:0;padding-bottom:0}.children.svelte-i6yrue{padding-left:var(--spacing-md)}.title.svelte-i6yrue{display:flex;align-items:center;color:var(--body-text-color);cursor:pointer;width:100%}.title.svelte-i6yrue .md{font-size:var(--text-sm) !important}.content.svelte-i6yrue{overflow-wrap:break-word;word-break:break-word;margin-left:var(--spacing-lg);margin-bottom:var(--spacing-sm)}.content.svelte-i6yrue *{font-size:var(--text-sm);color:var(--body-text-color)}.thought-group.svelte-i6yrue .thought:not(.nested){border:none;background:none}.duration.svelte-i6yrue{color:var(--body-text-color-subdued);font-size:var(--text-sm);margin-left:var(--size-1)}.arrow.svelte-i6yrue{opacity:0.8;width:var(--size-8);height:var(--size-8);display:flex;align-items:center;justify-content:center}.arrow.svelte-i6yrue button{background-color:transparent}.loading-spinner.svelte-i6yrue{display:inline-block;width:12px;height:12px;border:2px solid var(--body-text-color);border-radius:50%;border-top-color:transparent;animation:svelte-i6yrue-spin 1s linear infinite;margin:0 var(--size-1) -1px var(--size-2);opacity:0.8}@keyframes svelte-i6yrue-spin{to{transform:rotate(360deg)}}.thought-group.svelte-i6yrue .message-content{opacity:0.8}",
  map: '{"version":3,"file":"Thought.svelte","sources":["Thought.svelte"],"sourcesContent":["<script lang=\\"ts\\">import MessageContent from \\"./MessageContent.svelte\\";\\nimport { DropdownCircularArrow } from \\"@gradio/icons\\";\\nimport { IconButton } from \\"@gradio/atoms\\";\\nimport { slide } from \\"svelte/transition\\";\\nimport { MarkdownCode as Markdown } from \\"@gradio/markdown-code\\";\\nexport let thought;\\nexport let rtl = false;\\nexport let sanitize_html;\\nexport let latex_delimiters;\\nexport let render_markdown;\\nexport let _components;\\nexport let upload;\\nexport let thought_index;\\nexport let target;\\nexport let root;\\nexport let theme_mode;\\nexport let _fetch;\\nexport let scroll;\\nexport let allow_file_downloads;\\nexport let display_consecutive_in_same_bubble;\\nexport let i18n;\\nexport let line_breaks;\\nfunction is_thought_node(msg) {\\n    return \\"children\\" in msg;\\n}\\nlet thought_node;\\n$: thought_node = {\\n    ...thought,\\n    children: is_thought_node(thought) ? thought.children : []\\n};\\nfunction toggleExpanded() {\\n    expanded = !expanded;\\n}\\n$: expanded = thought_node.metadata?.status !== \\"done\\";\\n<\/script>\\n\\n<div class=\\"thought-group\\">\\n\\t<div\\n\\t\\tclass=\\"title\\"\\n\\t\\tclass:expanded\\n\\t\\ton:click|stopPropagation={toggleExpanded}\\n\\t\\taria-busy={thought_node.content === \\"\\" || thought_node.content === null}\\n\\t\\trole=\\"button\\"\\n\\t\\ttabindex=\\"0\\"\\n\\t\\ton:keydown={(e) => e.key === \\"Enter\\" && toggleExpanded()}\\n\\t>\\n\\t\\t<span\\n\\t\\t\\tclass=\\"arrow\\"\\n\\t\\t\\tstyle:transform={expanded ? \\"rotate(180deg)\\" : \\"rotate(0deg)\\"}\\n\\t\\t>\\n\\t\\t\\t<IconButton Icon={DropdownCircularArrow} />\\n\\t\\t</span>\\n\\t\\t<Markdown\\n\\t\\t\\tmessage={thought_node.metadata?.title || \\"\\"}\\n\\t\\t\\t{render_markdown}\\n\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t{sanitize_html}\\n\\t\\t\\t{root}\\n\\t\\t/>\\n\\t\\t{#if thought_node.metadata?.status === \\"pending\\"}\\n\\t\\t\\t<span class=\\"loading-spinner\\"></span>\\n\\t\\t{/if}\\n\\t\\t{#if thought_node?.metadata?.log || thought_node?.metadata?.duration}\\n\\t\\t\\t<span class=\\"duration\\">\\n\\t\\t\\t\\t{#if thought_node.metadata.log}\\n\\t\\t\\t\\t\\t{thought_node.metadata.log}\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t{#if thought_node.metadata.duration !== undefined}\\n\\t\\t\\t\\t\\t({#if Number.isInteger(thought_node.metadata.duration)}{thought_node\\n\\t\\t\\t\\t\\t\\t\\t.metadata\\n\\t\\t\\t\\t\\t\\t\\t.duration}s{:else if thought_node.metadata.duration >= 0.1}{thought_node.metadata.duration.toFixed(\\n\\t\\t\\t\\t\\t\\t\\t1\\n\\t\\t\\t\\t\\t\\t)}s{:else}{(thought_node.metadata.duration * 1000).toFixed(\\n\\t\\t\\t\\t\\t\\t\\t1\\n\\t\\t\\t\\t\\t\\t)}ms{/if})\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</span>\\n\\t\\t{/if}\\n\\t</div>\\n\\n\\t{#if expanded}\\n\\t\\t<div class=\\"content\\" transition:slide>\\n\\t\\t\\t<MessageContent\\n\\t\\t\\t\\tmessage={thought_node}\\n\\t\\t\\t\\t{sanitize_html}\\n\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t{render_markdown}\\n\\t\\t\\t\\t{_components}\\n\\t\\t\\t\\t{upload}\\n\\t\\t\\t\\t{thought_index}\\n\\t\\t\\t\\t{target}\\n\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t{theme_mode}\\n\\t\\t\\t\\t{_fetch}\\n\\t\\t\\t\\t{scroll}\\n\\t\\t\\t\\t{allow_file_downloads}\\n\\t\\t\\t\\t{display_consecutive_in_same_bubble}\\n\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t/>\\n\\n\\t\\t\\t{#if thought_node.children?.length > 0}\\n\\t\\t\\t\\t<div class=\\"children\\">\\n\\t\\t\\t\\t\\t{#each thought_node.children as child, index}\\n\\t\\t\\t\\t\\t\\t<svelte:self\\n\\t\\t\\t\\t\\t\\t\\tthought={child}\\n\\t\\t\\t\\t\\t\\t\\t{rtl}\\n\\t\\t\\t\\t\\t\\t\\t{sanitize_html}\\n\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t{render_markdown}\\n\\t\\t\\t\\t\\t\\t\\t{_components}\\n\\t\\t\\t\\t\\t\\t\\t{upload}\\n\\t\\t\\t\\t\\t\\t\\tthought_index={thought_index + 1}\\n\\t\\t\\t\\t\\t\\t\\t{target}\\n\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t{theme_mode}\\n\\t\\t\\t\\t\\t\\t\\t{_fetch}\\n\\t\\t\\t\\t\\t\\t\\t{scroll}\\n\\t\\t\\t\\t\\t\\t\\t{allow_file_downloads}\\n\\t\\t\\t\\t\\t\\t\\t{display_consecutive_in_same_bubble}\\n\\t\\t\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t</div>\\n\\t\\t\\t{/if}\\n\\t\\t</div>\\n\\t{/if}\\n</div>\\n\\n<style>\\n\\t.thought-group {\\n\\t\\tbackground: var(--background-fill-primary);\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tpadding: var(--spacing-md);\\n\\t\\tmargin: var(--spacing-md) 0;\\n\\t\\tfont-size: var(--text-sm);\\n\\t}\\n\\n\\t.children :global(.thought-group) {\\n\\t\\tborder: none;\\n\\t\\tmargin: 0;\\n\\t\\tpadding-bottom: 0;\\n\\t}\\n\\n\\t.children {\\n\\t\\tpadding-left: var(--spacing-md);\\n\\t}\\n\\n\\t.title {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tcursor: pointer;\\n\\t\\twidth: 100%;\\n\\t}\\n\\n\\t.title :global(.md) {\\n\\t\\tfont-size: var(--text-sm) !important;\\n\\t}\\n\\n\\t.content {\\n\\t\\toverflow-wrap: break-word;\\n\\t\\tword-break: break-word;\\n\\t\\tmargin-left: var(--spacing-lg);\\n\\t\\tmargin-bottom: var(--spacing-sm);\\n\\t}\\n\\t.content :global(*) {\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.thought-group :global(.thought:not(.nested)) {\\n\\t\\tborder: none;\\n\\t\\tbackground: none;\\n\\t}\\n\\n\\t.duration {\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tmargin-left: var(--size-1);\\n\\t}\\n\\n\\t.arrow {\\n\\t\\topacity: 0.8;\\n\\t\\twidth: var(--size-8);\\n\\t\\theight: var(--size-8);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.arrow :global(button) {\\n\\t\\tbackground-color: transparent;\\n\\t}\\n\\n\\t.loading-spinner {\\n\\t\\tdisplay: inline-block;\\n\\t\\twidth: 12px;\\n\\t\\theight: 12px;\\n\\t\\tborder: 2px solid var(--body-text-color);\\n\\t\\tborder-radius: 50%;\\n\\t\\tborder-top-color: transparent;\\n\\t\\tanimation: spin 1s linear infinite;\\n\\t\\tmargin: 0 var(--size-1) -1px var(--size-2);\\n\\t\\topacity: 0.8;\\n\\t}\\n\\n\\t@keyframes spin {\\n\\t\\tto {\\n\\t\\t\\ttransform: rotate(360deg);\\n\\t\\t}\\n\\t}\\n\\n\\t.thought-group :global(.message-content) {\\n\\t\\topacity: 0.8;\\n\\t}</style>\\n"],"names":[],"mappings":"AAmIC,4BAAe,CACd,UAAU,CAAE,IAAI,yBAAyB,CAAC,CAC1C,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CAAC,CAC7C,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,OAAO,CAAE,IAAI,YAAY,CAAC,CAC1B,MAAM,CAAE,IAAI,YAAY,CAAC,CAAC,CAAC,CAC3B,SAAS,CAAE,IAAI,SAAS,CACzB,CAEA,uBAAS,CAAS,cAAgB,CACjC,MAAM,CAAE,IAAI,CACZ,MAAM,CAAE,CAAC,CACT,cAAc,CAAE,CACjB,CAEA,uBAAU,CACT,YAAY,CAAE,IAAI,YAAY,CAC/B,CAEA,oBAAO,CACN,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,MAAM,CAAE,OAAO,CACf,KAAK,CAAE,IACR,CAEA,oBAAM,CAAS,GAAK,CACnB,SAAS,CAAE,IAAI,SAAS,CAAC,CAAC,UAC3B,CAEA,sBAAS,CACR,aAAa,CAAE,UAAU,CACzB,UAAU,CAAE,UAAU,CACtB,WAAW,CAAE,IAAI,YAAY,CAAC,CAC9B,aAAa,CAAE,IAAI,YAAY,CAChC,CACA,sBAAQ,CAAS,CAAG,CACnB,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,4BAAc,CAAS,qBAAuB,CAC7C,MAAM,CAAE,IAAI,CACZ,UAAU,CAAE,IACb,CAEA,uBAAU,CACT,KAAK,CAAE,IAAI,yBAAyB,CAAC,CACrC,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,WAAW,CAAE,IAAI,QAAQ,CAC1B,CAEA,oBAAO,CACN,OAAO,CAAE,GAAG,CACZ,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAClB,CAEA,oBAAM,CAAS,MAAQ,CACtB,gBAAgB,CAAE,WACnB,CAEA,8BAAiB,CAChB,OAAO,CAAE,YAAY,CACrB,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,iBAAiB,CAAC,CACxC,aAAa,CAAE,GAAG,CAClB,gBAAgB,CAAE,WAAW,CAC7B,SAAS,CAAE,kBAAI,CAAC,EAAE,CAAC,MAAM,CAAC,QAAQ,CAClC,MAAM,CAAE,CAAC,CAAC,IAAI,QAAQ,CAAC,CAAC,IAAI,CAAC,IAAI,QAAQ,CAAC,CAC1C,OAAO,CAAE,GACV,CAEA,WAAW,kBAAK,CACf,EAAG,CACF,SAAS,CAAE,OAAO,MAAM,CACzB,CACD,CAEA,4BAAc,CAAS,gBAAkB,CACxC,OAAO,CAAE,GACV"}'
};
function is_thought_node(msg) {
  return "children" in msg;
}
const Thought = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let expanded;
  let { thought } = $$props;
  let { rtl = false } = $$props;
  let { sanitize_html } = $$props;
  let { latex_delimiters } = $$props;
  let { render_markdown } = $$props;
  let { _components } = $$props;
  let { upload } = $$props;
  let { thought_index } = $$props;
  let { target } = $$props;
  let { root } = $$props;
  let { theme_mode } = $$props;
  let { _fetch } = $$props;
  let { scroll: scroll2 } = $$props;
  let { allow_file_downloads } = $$props;
  let { display_consecutive_in_same_bubble } = $$props;
  let { i18n } = $$props;
  let { line_breaks } = $$props;
  let thought_node;
  if ($$props.thought === void 0 && $$bindings.thought && thought !== void 0)
    $$bindings.thought(thought);
  if ($$props.rtl === void 0 && $$bindings.rtl && rtl !== void 0)
    $$bindings.rtl(rtl);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.render_markdown === void 0 && $$bindings.render_markdown && render_markdown !== void 0)
    $$bindings.render_markdown(render_markdown);
  if ($$props._components === void 0 && $$bindings._components && _components !== void 0)
    $$bindings._components(_components);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.thought_index === void 0 && $$bindings.thought_index && thought_index !== void 0)
    $$bindings.thought_index(thought_index);
  if ($$props.target === void 0 && $$bindings.target && target !== void 0)
    $$bindings.target(target);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props._fetch === void 0 && $$bindings._fetch && _fetch !== void 0)
    $$bindings._fetch(_fetch);
  if ($$props.scroll === void 0 && $$bindings.scroll && scroll2 !== void 0)
    $$bindings.scroll(scroll2);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  if ($$props.display_consecutive_in_same_bubble === void 0 && $$bindings.display_consecutive_in_same_bubble && display_consecutive_in_same_bubble !== void 0)
    $$bindings.display_consecutive_in_same_bubble(display_consecutive_in_same_bubble);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  $$result.css.add(css$5);
  thought_node = {
    ...thought,
    children: is_thought_node(thought) ? thought.children : []
  };
  expanded = thought_node.metadata?.status !== "done";
  return `<div class="thought-group svelte-i6yrue"><div class="${["title svelte-i6yrue", expanded ? "expanded" : ""].join(" ").trim()}"${add_attribute("aria-busy", thought_node.content === "" || thought_node.content === null, 0)} role="button" tabindex="0"><span class="arrow svelte-i6yrue"${add_styles({
    "transform": expanded ? "rotate(180deg)" : "rotate(0deg)"
  })}>${validate_component(IconButton, "IconButton").$$render($$result, { Icon: DropdownCircularArrow }, {}, {})}</span> ${validate_component(MarkdownCode, "Markdown").$$render(
    $$result,
    {
      message: thought_node.metadata?.title || "",
      render_markdown,
      latex_delimiters,
      sanitize_html,
      root
    },
    {},
    {}
  )} ${thought_node.metadata?.status === "pending" ? `<span class="loading-spinner svelte-i6yrue"></span>` : ``} ${thought_node?.metadata?.log || thought_node?.metadata?.duration ? `<span class="duration svelte-i6yrue">${thought_node.metadata.log ? `${escape(thought_node.metadata.log)}` : ``} ${thought_node.metadata.duration !== void 0 ? `(${Number.isInteger(thought_node.metadata.duration) ? `${escape(thought_node.metadata.duration)}s` : `${thought_node.metadata.duration >= 0.1 ? `${escape(thought_node.metadata.duration.toFixed(1))}s` : `${escape((thought_node.metadata.duration * 1e3).toFixed(1))}ms`}`})` : ``}</span>` : ``}</div> ${expanded ? `<div class="content svelte-i6yrue">${validate_component(MessageContent, "MessageContent").$$render(
    $$result,
    {
      message: thought_node,
      sanitize_html,
      latex_delimiters,
      render_markdown,
      _components,
      upload,
      thought_index,
      target,
      root,
      theme_mode,
      _fetch,
      scroll: scroll2,
      allow_file_downloads,
      display_consecutive_in_same_bubble,
      i18n,
      line_breaks
    },
    {},
    {}
  )} ${thought_node.children?.length > 0 ? `<div class="children svelte-i6yrue">${each(thought_node.children, (child, index) => {
    return `${validate_component(Thought, "svelte:self").$$render(
      $$result,
      {
        thought: child,
        rtl,
        sanitize_html,
        latex_delimiters,
        render_markdown,
        _components,
        upload,
        thought_index: thought_index + 1,
        target,
        root,
        theme_mode,
        _fetch,
        scroll: scroll2,
        allow_file_downloads,
        display_consecutive_in_same_bubble,
        i18n,
        line_breaks
      },
      {},
      {}
    )}`;
  })}</div>` : ``}</div>` : ``} </div>`;
});
const css$4 = {
  code: ".message.svelte-pcjl1g.svelte-pcjl1g{position:relative;width:100%;margin-top:var(--spacing-sm)}.message.display_consecutive_in_same_bubble.svelte-pcjl1g.svelte-pcjl1g{margin-top:0}.avatar-container.svelte-pcjl1g.svelte-pcjl1g{flex-shrink:0;border-radius:50%;border:1px solid var(--border-color-primary);overflow:hidden}.avatar-container.svelte-pcjl1g img{object-fit:cover}.flex-wrap.svelte-pcjl1g.svelte-pcjl1g{display:flex;flex-direction:column;width:calc(100% - var(--spacing-xxl));max-width:100%;color:var(--body-text-color);font-size:var(--chatbot-text-size);overflow-wrap:break-word;width:100%;height:100%}.component.svelte-pcjl1g.svelte-pcjl1g{padding:0;border-radius:var(--radius-md);width:fit-content;overflow:hidden}.component.gallery.svelte-pcjl1g.svelte-pcjl1g{border:none}.message-row.svelte-pcjl1g .svelte-pcjl1g:not(.avatar-container) img{margin:var(--size-2);max-height:300px}.file-pil.svelte-pcjl1g.svelte-pcjl1g{display:block;width:fit-content;padding:var(--spacing-sm) var(--spacing-lg);border-radius:var(--radius-md);background:var(--background-fill-secondary);color:var(--body-text-color);text-decoration:none;margin:0;font-family:var(--font-mono);font-size:var(--text-sm)}.file.svelte-pcjl1g.svelte-pcjl1g{width:auto !important;max-width:fit-content !important}@media(max-width: 600px) or (max-width: 480px){.component.svelte-pcjl1g.svelte-pcjl1g{width:100%}}.message.svelte-pcjl1g .prose{font-size:var(--chatbot-text-size)}.message-bubble-border.svelte-pcjl1g.svelte-pcjl1g{border-width:1px;border-radius:var(--radius-md)}.panel-full-width.svelte-pcjl1g.svelte-pcjl1g{width:100%}.message-markdown-disabled.svelte-pcjl1g.svelte-pcjl1g{white-space:pre-line}.user.svelte-pcjl1g.svelte-pcjl1g{border-radius:var(--radius-md);align-self:flex-end;border-bottom-right-radius:0;box-shadow:var(--shadow-drop);border:1px solid var(--border-color-accent-subdued);background-color:var(--color-accent-soft);padding:var(--spacing-sm) var(--spacing-xl)}.bot.svelte-pcjl1g.svelte-pcjl1g{border:1px solid var(--border-color-primary);border-radius:var(--radius-md);border-color:var(--border-color-primary);background-color:var(--background-fill-secondary);box-shadow:var(--shadow-drop);align-self:flex-start;text-align:right;border-bottom-left-radius:0;padding:var(--spacing-sm) var(--spacing-xl)}.bot.svelte-pcjl1g.svelte-pcjl1g:has(.table-wrap){border:none;box-shadow:none;background:none}.panel.svelte-pcjl1g .user.svelte-pcjl1g *{text-align:right}.message-row.svelte-pcjl1g.svelte-pcjl1g{display:flex;position:relative}.bubble.svelte-pcjl1g.svelte-pcjl1g{margin:calc(var(--spacing-xl) * 2);margin-bottom:var(--spacing-xl)}.bubble.user-row.svelte-pcjl1g.svelte-pcjl1g{align-self:flex-end;max-width:calc(100% - var(--spacing-xl) * 6)}.bubble.bot-row.svelte-pcjl1g.svelte-pcjl1g{align-self:flex-start;max-width:calc(100% - var(--spacing-xl) * 6)}.bubble.svelte-pcjl1g .user-row.svelte-pcjl1g{flex-direction:row;justify-content:flex-end}.bubble.svelte-pcjl1g .with_avatar.user-row.svelte-pcjl1g{margin-right:calc(var(--spacing-xl) * 2) !important}.bubble.svelte-pcjl1g .with_avatar.bot-row.svelte-pcjl1g{margin-left:calc(var(--spacing-xl) * 2) !important}.bubble.svelte-pcjl1g .with_opposite_avatar.user-row.svelte-pcjl1g{margin-left:calc(var(--spacing-xxl) + 35px + var(--spacing-xxl))}.panel.svelte-pcjl1g.svelte-pcjl1g{margin:0;padding:calc(var(--spacing-lg) * 2) calc(var(--spacing-lg) * 2)}.panel.bot-row.svelte-pcjl1g.svelte-pcjl1g{background:var(--background-fill-secondary)}.panel.svelte-pcjl1g .with_avatar.svelte-pcjl1g{padding-left:calc(var(--spacing-xl) * 2) !important;padding-right:calc(var(--spacing-xl) * 2) !important}.panel.svelte-pcjl1g .panel-full-width.svelte-pcjl1g{width:100%}.panel.svelte-pcjl1g .user.svelte-pcjl1g *{text-align:right}.flex-wrap.svelte-pcjl1g.svelte-pcjl1g{display:flex;flex-direction:column;max-width:100%;color:var(--body-text-color);font-size:var(--chatbot-text-size);overflow-wrap:break-word}@media(max-width: 480px){.user-row.bubble.svelte-pcjl1g.svelte-pcjl1g{align-self:flex-end}.bot-row.bubble.svelte-pcjl1g.svelte-pcjl1g{align-self:flex-start}.message.svelte-pcjl1g.svelte-pcjl1g{width:100%}}.avatar-container.svelte-pcjl1g.svelte-pcjl1g{align-self:flex-start;position:relative;display:flex;justify-content:flex-start;align-items:flex-start;width:35px;height:35px;flex-shrink:0;bottom:0;border-radius:50%;border:1px solid var(--border-color-primary)}.user-row.svelte-pcjl1g>.avatar-container.svelte-pcjl1g{order:2}.user-row.bubble.svelte-pcjl1g>.avatar-container.svelte-pcjl1g{margin-left:var(--spacing-xxl)}.bot-row.bubble.svelte-pcjl1g>.avatar-container.svelte-pcjl1g{margin-left:var(--spacing-xxl)}.panel.user-row.svelte-pcjl1g>.avatar-container.svelte-pcjl1g{order:0}.bot-row.bubble.svelte-pcjl1g>.avatar-container.svelte-pcjl1g{margin-right:var(--spacing-xxl);margin-left:0}.avatar-container.svelte-pcjl1g:not(.thumbnail-item) img{width:100%;height:100%;object-fit:cover;border-radius:50%;padding:6px}.selectable.svelte-pcjl1g.svelte-pcjl1g{cursor:pointer}@keyframes svelte-pcjl1g-dot-flashing{0%{opacity:0.8}50%{opacity:0.5}100%{opacity:0.8}}.message.svelte-pcjl1g .preview{object-fit:contain;width:95%;max-height:93%}.image-preview.svelte-pcjl1g.svelte-pcjl1g{position:absolute;z-index:999;left:0;top:0;width:100%;height:100%;overflow:auto;background-color:rgba(0, 0, 0, 0.9);display:flex;justify-content:center;align-items:center}.image-preview.svelte-pcjl1g svg{stroke:white}.image-preview-close-button.svelte-pcjl1g.svelte-pcjl1g{position:absolute;top:10px;right:10px;background:none;border:none;font-size:1.5em;cursor:pointer;height:30px;width:30px;padding:3px;background:var(--bg-color);box-shadow:var(--shadow-drop);border:1px solid var(--button-secondary-border-color);border-radius:var(--radius-lg)}.message.svelte-pcjl1g>div.svelte-pcjl1g{width:100%}.html.svelte-pcjl1g.svelte-pcjl1g{padding:0;border:none;background:none}.panel.svelte-pcjl1g .bot.svelte-pcjl1g,.panel.svelte-pcjl1g .user.svelte-pcjl1g{border:none;box-shadow:none;background-color:var(--background-fill-secondary)}textarea.svelte-pcjl1g.svelte-pcjl1g{background:none;border-radius:var(--radius-lg);border:none;display:block;max-width:100%}.user.svelte-pcjl1g textarea.svelte-pcjl1g{border-bottom-right-radius:0}.bot.svelte-pcjl1g textarea.svelte-pcjl1g{border-bottom-left-radius:0}.user.svelte-pcjl1g textarea.svelte-pcjl1g:focus{outline:2px solid var(--border-color-accent)}.bot.svelte-pcjl1g textarea.svelte-pcjl1g:focus{outline:2px solid var(--border-color-primary)}.panel.user-row.svelte-pcjl1g.svelte-pcjl1g{background-color:var(--color-accent-soft)}.panel.svelte-pcjl1g .user-row.svelte-pcjl1g,.panel.svelte-pcjl1g .bot-row.svelte-pcjl1g{align-self:flex-start}.panel.svelte-pcjl1g .user.svelte-pcjl1g *,.panel.svelte-pcjl1g .bot.svelte-pcjl1g *{text-align:left}.panel.svelte-pcjl1g .user.svelte-pcjl1g{background-color:var(--color-accent-soft)}.panel.svelte-pcjl1g .user-row.svelte-pcjl1g{background-color:var(--color-accent-soft);align-self:flex-start}.panel.svelte-pcjl1g .message.svelte-pcjl1g{margin-bottom:var(--spacing-md)}",
  map: '{"version":3,"file":"Message.svelte","sources":["Message.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { is_component_message } from \\"../shared/utils\\";\\nimport { Image } from \\"@gradio/image/shared\\";\\nimport ButtonPanel from \\"./ButtonPanel.svelte\\";\\nimport MessageContent from \\"./MessageContent.svelte\\";\\nimport Thought from \\"./Thought.svelte\\";\\nexport let value;\\nexport let avatar_img;\\nexport let opposite_avatar_img = null;\\nexport let role = \\"user\\";\\nexport let messages = [];\\nexport let layout;\\nexport let render_markdown;\\nexport let latex_delimiters;\\nexport let sanitize_html;\\nexport let selectable;\\nexport let _fetch;\\nexport let rtl;\\nexport let dispatch;\\nexport let i18n;\\nexport let line_breaks;\\nexport let upload;\\nexport let target;\\nexport let root;\\nexport let theme_mode;\\nexport let _components;\\nexport let i;\\nexport let show_copy_button;\\nexport let generating;\\nexport let feedback_options;\\nexport let show_like;\\nexport let show_edit;\\nexport let show_retry;\\nexport let show_undo;\\nexport let msg_format;\\nexport let handle_action;\\nexport let scroll;\\nexport let allow_file_downloads;\\nexport let in_edit_mode;\\nexport let edit_message;\\nexport let display_consecutive_in_same_bubble;\\nexport let current_feedback = null;\\nlet messageElements = [];\\nlet previous_edit_mode = false;\\nlet last_message_width = 0;\\nlet last_message_height = 0;\\n$: if (in_edit_mode && !previous_edit_mode) {\\n    last_message_width = messageElements[messageElements.length - 1]?.clientWidth;\\n    last_message_height = messageElements[messageElements.length - 1]?.clientHeight;\\n}\\nfunction handle_select(i2, message) {\\n    dispatch(\\"select\\", {\\n        index: message.index,\\n        value: message.content\\n    });\\n}\\nfunction get_message_label_data(message) {\\n    if (message.type === \\"text\\") {\\n        return message.content;\\n    }\\n    else if (message.type === \\"component\\" && message.content.component === \\"file\\") {\\n        if (Array.isArray(message.content.value)) {\\n            return `file of extension type: ${message.content.value[0].orig_name?.split(\\".\\").pop()}`;\\n        }\\n        return `file of extension type: ${message.content.value?.orig_name?.split(\\".\\").pop()}` + (message.content.value?.orig_name ?? \\"\\");\\n    }\\n    return `a component of type ${message.content.component ?? \\"unknown\\"}`;\\n}\\nlet button_panel_props;\\n$: button_panel_props = {\\n    handle_action,\\n    likeable: show_like,\\n    feedback_options,\\n    show_retry,\\n    show_undo,\\n    show_edit,\\n    in_edit_mode,\\n    generating,\\n    show_copy_button,\\n    message: msg_format === \\"tuples\\" ? messages[0] : messages,\\n    position: role === \\"user\\" ? \\"right\\" : \\"left\\",\\n    avatar: avatar_img,\\n    layout,\\n    dispatch,\\n    current_feedback\\n};\\n<\/script>\\n\\n<div\\n\\tclass=\\"message-row {layout} {role}-row\\"\\n\\tclass:with_avatar={avatar_img !== null}\\n\\tclass:with_opposite_avatar={opposite_avatar_img !== null}\\n>\\n\\t{#if avatar_img !== null}\\n\\t\\t<div class=\\"avatar-container\\">\\n\\t\\t\\t<Image class=\\"avatar-image\\" src={avatar_img?.url} alt=\\"{role} avatar\\" />\\n\\t\\t</div>\\n\\t{/if}\\n\\t<div\\n\\t\\tclass:role\\n\\t\\tclass=\\"flex-wrap\\"\\n\\t\\tclass:component-wrap={messages[0].type === \\"component\\"}\\n\\t>\\n\\t\\t<div\\n\\t\\t\\tclass:message={display_consecutive_in_same_bubble}\\n\\t\\t\\tclass={display_consecutive_in_same_bubble ? role : \\"\\"}\\n\\t\\t>\\n\\t\\t\\t{#each messages as message, thought_index}\\n\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\tclass=\\"message {!display_consecutive_in_same_bubble ? role : \'\'}\\"\\n\\t\\t\\t\\t\\tclass:panel-full-width={true}\\n\\t\\t\\t\\t\\tclass:message-markdown-disabled={!render_markdown}\\n\\t\\t\\t\\t\\tclass:component={message.type === \\"component\\"}\\n\\t\\t\\t\\t\\tclass:html={is_component_message(message) &&\\n\\t\\t\\t\\t\\t\\tmessage.content.component === \\"html\\"}\\n\\t\\t\\t\\t\\tclass:thought={thought_index > 0}\\n\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t{#if in_edit_mode && thought_index === messages.length - 1 && message.type === \\"text\\"}\\n\\t\\t\\t\\t\\t\\t<!-- svelte-ignore a11y-autofocus -->\\n\\t\\t\\t\\t\\t\\t<textarea\\n\\t\\t\\t\\t\\t\\t\\tclass=\\"edit-textarea\\"\\n\\t\\t\\t\\t\\t\\t\\tstyle:width={`max(${last_message_width}px, 160px)`}\\n\\t\\t\\t\\t\\t\\t\\tstyle:min-height={`${last_message_height}px`}\\n\\t\\t\\t\\t\\t\\t\\tautofocus\\n\\t\\t\\t\\t\\t\\t\\tbind:value={edit_message}\\n\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t<!-- svelte-ignore a11y-no-static-element-interactions -->\\n\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\tdata-testid={role}\\n\\t\\t\\t\\t\\t\\t\\tclass:latest={i === value.length - 1}\\n\\t\\t\\t\\t\\t\\t\\tclass:message-markdown-disabled={!render_markdown}\\n\\t\\t\\t\\t\\t\\t\\tstyle:user-select=\\"text\\"\\n\\t\\t\\t\\t\\t\\t\\tclass:selectable\\n\\t\\t\\t\\t\\t\\t\\tstyle:cursor={selectable ? \\"pointer\\" : \\"auto\\"}\\n\\t\\t\\t\\t\\t\\t\\tstyle:text-align={rtl ? \\"right\\" : \\"left\\"}\\n\\t\\t\\t\\t\\t\\t\\tbind:this={messageElements[thought_index]}\\n\\t\\t\\t\\t\\t\\t\\ton:click={() => handle_select(i, message)}\\n\\t\\t\\t\\t\\t\\t\\ton:keydown={(e) => {\\n\\t\\t\\t\\t\\t\\t\\t\\tif (e.key === \\"Enter\\") {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\thandle_select(i, message);\\n\\t\\t\\t\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\tdir={rtl ? \\"rtl\\" : \\"ltr\\"}\\n\\t\\t\\t\\t\\t\\t\\taria-label={role +\\n\\t\\t\\t\\t\\t\\t\\t\\t\\"\'s message: \\" +\\n\\t\\t\\t\\t\\t\\t\\t\\tget_message_label_data(message)}\\n\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t{#if message?.metadata?.title}\\n\\t\\t\\t\\t\\t\\t\\t\\t<Thought\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tthought={message}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{rtl}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{sanitize_html}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{render_markdown}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{_components}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{upload}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{thought_index}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{target}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{theme_mode}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{_fetch}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{scroll}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{allow_file_downloads}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{display_consecutive_in_same_bubble}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t\\t\\t<MessageContent\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{message}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{sanitize_html}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{render_markdown}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{_components}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{upload}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{thought_index}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{target}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{theme_mode}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{_fetch}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{scroll}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{allow_file_downloads}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{display_consecutive_in_same_bubble}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t</div>\\n\\n\\t\\t\\t\\t{#if layout === \\"panel\\"}\\n\\t\\t\\t\\t\\t<ButtonPanel\\n\\t\\t\\t\\t\\t\\t{...button_panel_props}\\n\\t\\t\\t\\t\\t\\t{current_feedback}\\n\\t\\t\\t\\t\\t\\ton:copy={(e) => dispatch(\\"copy\\", e.detail)}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t{/each}\\n\\t\\t</div>\\n\\t</div>\\n</div>\\n\\n{#if layout === \\"bubble\\"}\\n\\t<ButtonPanel {...button_panel_props} />\\n{/if}\\n\\n<style>\\n\\t.message {\\n\\t\\tposition: relative;\\n\\t\\twidth: 100%;\\n\\t\\tmargin-top: var(--spacing-sm);\\n\\t}\\n\\n\\t.message.display_consecutive_in_same_bubble {\\n\\t\\tmargin-top: 0;\\n\\t}\\n\\n\\t/* avatar styles */\\n\\t.avatar-container {\\n\\t\\tflex-shrink: 0;\\n\\t\\tborder-radius: 50%;\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\toverflow: hidden;\\n\\t}\\n\\n\\t.avatar-container :global(img) {\\n\\t\\tobject-fit: cover;\\n\\t}\\n\\n\\t/* message wrapper */\\n\\t.flex-wrap {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\twidth: calc(100% - var(--spacing-xxl));\\n\\t\\tmax-width: 100%;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-size: var(--chatbot-text-size);\\n\\t\\toverflow-wrap: break-word;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t}\\n\\n\\t.component {\\n\\t\\tpadding: 0;\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\twidth: fit-content;\\n\\t\\toverflow: hidden;\\n\\t}\\n\\n\\t.component.gallery {\\n\\t\\tborder: none;\\n\\t}\\n\\n\\t.message-row :not(.avatar-container) :global(img) {\\n\\t\\tmargin: var(--size-2);\\n\\t\\tmax-height: 300px;\\n\\t}\\n\\n\\t.file-pil {\\n\\t\\tdisplay: block;\\n\\t\\twidth: fit-content;\\n\\t\\tpadding: var(--spacing-sm) var(--spacing-lg);\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\ttext-decoration: none;\\n\\t\\tmargin: 0;\\n\\t\\tfont-family: var(--font-mono);\\n\\t\\tfont-size: var(--text-sm);\\n\\t}\\n\\n\\t.file {\\n\\t\\twidth: auto !important;\\n\\t\\tmax-width: fit-content !important;\\n\\t}\\n\\n\\t@media (max-width: 600px) or (max-width: 480px) {\\n\\t\\t.component {\\n\\t\\t\\twidth: 100%;\\n\\t\\t}\\n\\t}\\n\\n\\t.message :global(.prose) {\\n\\t\\tfont-size: var(--chatbot-text-size);\\n\\t}\\n\\n\\t.message-bubble-border {\\n\\t\\tborder-width: 1px;\\n\\t\\tborder-radius: var(--radius-md);\\n\\t}\\n\\n\\t.panel-full-width {\\n\\t\\twidth: 100%;\\n\\t}\\n\\t.message-markdown-disabled {\\n\\t\\twhite-space: pre-line;\\n\\t}\\n\\n\\t.user {\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\talign-self: flex-end;\\n\\t\\tborder-bottom-right-radius: 0;\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t\\tborder: 1px solid var(--border-color-accent-subdued);\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t\\tpadding: var(--spacing-sm) var(--spacing-xl);\\n\\t}\\n\\n\\t.bot {\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tborder-color: var(--border-color-primary);\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t\\talign-self: flex-start;\\n\\t\\ttext-align: right;\\n\\t\\tborder-bottom-left-radius: 0;\\n\\t\\tpadding: var(--spacing-sm) var(--spacing-xl);\\n\\t}\\n\\n\\t.bot:has(.table-wrap) {\\n\\t\\tborder: none;\\n\\t\\tbox-shadow: none;\\n\\t\\tbackground: none;\\n\\t}\\n\\n\\t.panel .user :global(*) {\\n\\t\\ttext-align: right;\\n\\t}\\n\\n\\t/* Colors */\\n\\n\\t.message-row {\\n\\t\\tdisplay: flex;\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t/* bubble mode styles */\\n\\t.bubble {\\n\\t\\tmargin: calc(var(--spacing-xl) * 2);\\n\\t\\tmargin-bottom: var(--spacing-xl);\\n\\t}\\n\\n\\t.bubble.user-row {\\n\\t\\talign-self: flex-end;\\n\\t\\tmax-width: calc(100% - var(--spacing-xl) * 6);\\n\\t}\\n\\n\\t.bubble.bot-row {\\n\\t\\talign-self: flex-start;\\n\\t\\tmax-width: calc(100% - var(--spacing-xl) * 6);\\n\\t}\\n\\n\\t.bubble .user-row {\\n\\t\\tflex-direction: row;\\n\\t\\tjustify-content: flex-end;\\n\\t}\\n\\n\\t.bubble .with_avatar.user-row {\\n\\t\\tmargin-right: calc(var(--spacing-xl) * 2) !important;\\n\\t}\\n\\n\\t.bubble .with_avatar.bot-row {\\n\\t\\tmargin-left: calc(var(--spacing-xl) * 2) !important;\\n\\t}\\n\\n\\t.bubble .with_opposite_avatar.user-row {\\n\\t\\tmargin-left: calc(var(--spacing-xxl) + 35px + var(--spacing-xxl));\\n\\t}\\n\\n\\t/* panel mode styles */\\n\\t.panel {\\n\\t\\tmargin: 0;\\n\\t\\tpadding: calc(var(--spacing-lg) * 2) calc(var(--spacing-lg) * 2);\\n\\t}\\n\\n\\t.panel.bot-row {\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t}\\n\\n\\t.panel .with_avatar {\\n\\t\\tpadding-left: calc(var(--spacing-xl) * 2) !important;\\n\\t\\tpadding-right: calc(var(--spacing-xl) * 2) !important;\\n\\t}\\n\\n\\t.panel .panel-full-width {\\n\\t\\twidth: 100%;\\n\\t}\\n\\n\\t.panel .user :global(*) {\\n\\t\\ttext-align: right;\\n\\t}\\n\\n\\t/* message content */\\n\\t.flex-wrap {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tmax-width: 100%;\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-size: var(--chatbot-text-size);\\n\\t\\toverflow-wrap: break-word;\\n\\t}\\n\\n\\t@media (max-width: 480px) {\\n\\t\\t.user-row.bubble {\\n\\t\\t\\talign-self: flex-end;\\n\\t\\t}\\n\\n\\t\\t.bot-row.bubble {\\n\\t\\t\\talign-self: flex-start;\\n\\t\\t}\\n\\t\\t.message {\\n\\t\\t\\twidth: 100%;\\n\\t\\t}\\n\\t}\\n\\n\\t.avatar-container {\\n\\t\\talign-self: flex-start;\\n\\t\\tposition: relative;\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: flex-start;\\n\\t\\talign-items: flex-start;\\n\\t\\twidth: 35px;\\n\\t\\theight: 35px;\\n\\t\\tflex-shrink: 0;\\n\\t\\tbottom: 0;\\n\\t\\tborder-radius: 50%;\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t}\\n\\t.user-row > .avatar-container {\\n\\t\\torder: 2;\\n\\t}\\n\\n\\t.user-row.bubble > .avatar-container {\\n\\t\\tmargin-left: var(--spacing-xxl);\\n\\t}\\n\\n\\t.bot-row.bubble > .avatar-container {\\n\\t\\tmargin-left: var(--spacing-xxl);\\n\\t}\\n\\n\\t.panel.user-row > .avatar-container {\\n\\t\\torder: 0;\\n\\t}\\n\\n\\t.bot-row.bubble > .avatar-container {\\n\\t\\tmargin-right: var(--spacing-xxl);\\n\\t\\tmargin-left: 0;\\n\\t}\\n\\n\\t.avatar-container:not(.thumbnail-item) :global(img) {\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\tobject-fit: cover;\\n\\t\\tborder-radius: 50%;\\n\\t\\tpadding: 6px;\\n\\t}\\n\\n\\t.selectable {\\n\\t\\tcursor: pointer;\\n\\t}\\n\\n\\t@keyframes dot-flashing {\\n\\t\\t0% {\\n\\t\\t\\topacity: 0.8;\\n\\t\\t}\\n\\t\\t50% {\\n\\t\\t\\topacity: 0.5;\\n\\t\\t}\\n\\t\\t100% {\\n\\t\\t\\topacity: 0.8;\\n\\t\\t}\\n\\t}\\n\\n\\t/* Image preview */\\n\\t.message :global(.preview) {\\n\\t\\tobject-fit: contain;\\n\\t\\twidth: 95%;\\n\\t\\tmax-height: 93%;\\n\\t}\\n\\t.image-preview {\\n\\t\\tposition: absolute;\\n\\t\\tz-index: 999;\\n\\t\\tleft: 0;\\n\\t\\ttop: 0;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\toverflow: auto;\\n\\t\\tbackground-color: rgba(0, 0, 0, 0.9);\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: center;\\n\\t\\talign-items: center;\\n\\t}\\n\\t.image-preview :global(svg) {\\n\\t\\tstroke: white;\\n\\t}\\n\\t.image-preview-close-button {\\n\\t\\tposition: absolute;\\n\\t\\ttop: 10px;\\n\\t\\tright: 10px;\\n\\t\\tbackground: none;\\n\\t\\tborder: none;\\n\\t\\tfont-size: 1.5em;\\n\\t\\tcursor: pointer;\\n\\t\\theight: 30px;\\n\\t\\twidth: 30px;\\n\\t\\tpadding: 3px;\\n\\t\\tbackground: var(--bg-color);\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t\\tborder: 1px solid var(--button-secondary-border-color);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t}\\n\\n\\t.message > div {\\n\\t\\twidth: 100%;\\n\\t}\\n\\t.html {\\n\\t\\tpadding: 0;\\n\\t\\tborder: none;\\n\\t\\tbackground: none;\\n\\t}\\n\\n\\t.panel .bot,\\n\\t.panel .user {\\n\\t\\tborder: none;\\n\\t\\tbox-shadow: none;\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t}\\n\\n\\ttextarea {\\n\\t\\tbackground: none;\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\tborder: none;\\n\\t\\tdisplay: block;\\n\\t\\tmax-width: 100%;\\n\\t}\\n\\t.user textarea {\\n\\t\\tborder-bottom-right-radius: 0;\\n\\t}\\n\\t.bot textarea {\\n\\t\\tborder-bottom-left-radius: 0;\\n\\t}\\n\\t.user textarea:focus {\\n\\t\\toutline: 2px solid var(--border-color-accent);\\n\\t}\\n\\t.bot textarea:focus {\\n\\t\\toutline: 2px solid var(--border-color-primary);\\n\\t}\\n\\n\\t.panel.user-row {\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t}\\n\\n\\t.panel .user-row,\\n\\t.panel .bot-row {\\n\\t\\talign-self: flex-start;\\n\\t}\\n\\n\\t.panel .user :global(*),\\n\\t.panel .bot :global(*) {\\n\\t\\ttext-align: left;\\n\\t}\\n\\n\\t.panel .user {\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t}\\n\\n\\t.panel .user-row {\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t\\talign-self: flex-start;\\n\\t}\\n\\n\\t.panel .message {\\n\\t\\tmargin-bottom: var(--spacing-md);\\n\\t}</style>\\n"],"names":[],"mappings":"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"}'
};
let previous_edit_mode = false;
function get_message_label_data(message) {
  if (message.type === "text") {
    return message.content;
  } else if (message.type === "component" && message.content.component === "file") {
    if (Array.isArray(message.content.value)) {
      return `file of extension type: ${message.content.value[0].orig_name?.split(".").pop()}`;
    }
    return `file of extension type: ${message.content.value?.orig_name?.split(".").pop()}` + (message.content.value?.orig_name ?? "");
  }
  return `a component of type ${message.content.component ?? "unknown"}`;
}
const Message = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { avatar_img } = $$props;
  let { opposite_avatar_img = null } = $$props;
  let { role = "user" } = $$props;
  let { messages = [] } = $$props;
  let { layout } = $$props;
  let { render_markdown } = $$props;
  let { latex_delimiters } = $$props;
  let { sanitize_html } = $$props;
  let { selectable } = $$props;
  let { _fetch } = $$props;
  let { rtl } = $$props;
  let { dispatch } = $$props;
  let { i18n } = $$props;
  let { line_breaks } = $$props;
  let { upload } = $$props;
  let { target } = $$props;
  let { root } = $$props;
  let { theme_mode } = $$props;
  let { _components } = $$props;
  let { i } = $$props;
  let { show_copy_button } = $$props;
  let { generating } = $$props;
  let { feedback_options } = $$props;
  let { show_like } = $$props;
  let { show_edit } = $$props;
  let { show_retry } = $$props;
  let { show_undo } = $$props;
  let { msg_format } = $$props;
  let { handle_action } = $$props;
  let { scroll: scroll2 } = $$props;
  let { allow_file_downloads } = $$props;
  let { in_edit_mode } = $$props;
  let { edit_message } = $$props;
  let { display_consecutive_in_same_bubble } = $$props;
  let { current_feedback = null } = $$props;
  let messageElements = [];
  let last_message_width = 0;
  let last_message_height = 0;
  let button_panel_props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.avatar_img === void 0 && $$bindings.avatar_img && avatar_img !== void 0)
    $$bindings.avatar_img(avatar_img);
  if ($$props.opposite_avatar_img === void 0 && $$bindings.opposite_avatar_img && opposite_avatar_img !== void 0)
    $$bindings.opposite_avatar_img(opposite_avatar_img);
  if ($$props.role === void 0 && $$bindings.role && role !== void 0)
    $$bindings.role(role);
  if ($$props.messages === void 0 && $$bindings.messages && messages !== void 0)
    $$bindings.messages(messages);
  if ($$props.layout === void 0 && $$bindings.layout && layout !== void 0)
    $$bindings.layout(layout);
  if ($$props.render_markdown === void 0 && $$bindings.render_markdown && render_markdown !== void 0)
    $$bindings.render_markdown(render_markdown);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props._fetch === void 0 && $$bindings._fetch && _fetch !== void 0)
    $$bindings._fetch(_fetch);
  if ($$props.rtl === void 0 && $$bindings.rtl && rtl !== void 0)
    $$bindings.rtl(rtl);
  if ($$props.dispatch === void 0 && $$bindings.dispatch && dispatch !== void 0)
    $$bindings.dispatch(dispatch);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.target === void 0 && $$bindings.target && target !== void 0)
    $$bindings.target(target);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props._components === void 0 && $$bindings._components && _components !== void 0)
    $$bindings._components(_components);
  if ($$props.i === void 0 && $$bindings.i && i !== void 0)
    $$bindings.i(i);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.generating === void 0 && $$bindings.generating && generating !== void 0)
    $$bindings.generating(generating);
  if ($$props.feedback_options === void 0 && $$bindings.feedback_options && feedback_options !== void 0)
    $$bindings.feedback_options(feedback_options);
  if ($$props.show_like === void 0 && $$bindings.show_like && show_like !== void 0)
    $$bindings.show_like(show_like);
  if ($$props.show_edit === void 0 && $$bindings.show_edit && show_edit !== void 0)
    $$bindings.show_edit(show_edit);
  if ($$props.show_retry === void 0 && $$bindings.show_retry && show_retry !== void 0)
    $$bindings.show_retry(show_retry);
  if ($$props.show_undo === void 0 && $$bindings.show_undo && show_undo !== void 0)
    $$bindings.show_undo(show_undo);
  if ($$props.msg_format === void 0 && $$bindings.msg_format && msg_format !== void 0)
    $$bindings.msg_format(msg_format);
  if ($$props.handle_action === void 0 && $$bindings.handle_action && handle_action !== void 0)
    $$bindings.handle_action(handle_action);
  if ($$props.scroll === void 0 && $$bindings.scroll && scroll2 !== void 0)
    $$bindings.scroll(scroll2);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  if ($$props.in_edit_mode === void 0 && $$bindings.in_edit_mode && in_edit_mode !== void 0)
    $$bindings.in_edit_mode(in_edit_mode);
  if ($$props.edit_message === void 0 && $$bindings.edit_message && edit_message !== void 0)
    $$bindings.edit_message(edit_message);
  if ($$props.display_consecutive_in_same_bubble === void 0 && $$bindings.display_consecutive_in_same_bubble && display_consecutive_in_same_bubble !== void 0)
    $$bindings.display_consecutive_in_same_bubble(display_consecutive_in_same_bubble);
  if ($$props.current_feedback === void 0 && $$bindings.current_feedback && current_feedback !== void 0)
    $$bindings.current_feedback(current_feedback);
  $$result.css.add(css$4);
  {
    if (in_edit_mode && !previous_edit_mode) {
      last_message_width = messageElements[messageElements.length - 1]?.clientWidth;
      last_message_height = messageElements[messageElements.length - 1]?.clientHeight;
    }
  }
  button_panel_props = {
    handle_action,
    likeable: show_like,
    feedback_options,
    show_retry,
    show_undo,
    show_edit,
    in_edit_mode,
    generating,
    show_copy_button,
    message: msg_format === "tuples" ? messages[0] : messages,
    position: role === "user" ? "right" : "left",
    avatar: avatar_img,
    layout,
    dispatch,
    current_feedback
  };
  return `<div class="${[
    "message-row " + escape(layout, true) + " " + escape(role, true) + "-row svelte-pcjl1g",
    (avatar_img !== null ? "with_avatar" : "") + " " + (opposite_avatar_img !== null ? "with_opposite_avatar" : "")
  ].join(" ").trim()}">${avatar_img !== null ? `<div class="avatar-container svelte-pcjl1g">${validate_component(Image$1, "Image").$$render(
    $$result,
    {
      class: "avatar-image",
      src: avatar_img?.url,
      alt: role + " avatar"
    },
    {},
    {}
  )}</div>` : ``} <div class="${[
    "flex-wrap svelte-pcjl1g",
    (role ? "role" : "") + " " + (messages[0].type === "component" ? "component-wrap" : "")
  ].join(" ").trim()}"><div class="${[
    escape(null_to_empty(display_consecutive_in_same_bubble ? role : ""), true) + " svelte-pcjl1g",
    display_consecutive_in_same_bubble ? "message" : ""
  ].join(" ").trim()}">${each(messages, (message, thought_index) => {
    return `<div class="${[
      "message " + escape(!display_consecutive_in_same_bubble ? role : "", true) + " svelte-pcjl1g",
      "panel-full-width " + (!render_markdown ? "message-markdown-disabled" : "") + " " + (message.type === "component" ? "component" : "") + " " + (is_component_message(message) && message.content.component === "html" ? "html" : "") + " " + (thought_index > 0 ? "thought" : "")
    ].join(" ").trim()}">${in_edit_mode && thought_index === messages.length - 1 && message.type === "text" ? ` <textarea class="edit-textarea svelte-pcjl1g" autofocus${add_styles({
      "width": `max(${last_message_width}px, 160px)`,
      "min-height": `${last_message_height}px`
    })}>${escape(edit_message || "")}</textarea>` : ` <div${add_attribute("data-testid", role, 0)}${add_attribute("dir", rtl ? "rtl" : "ltr", 0)}${add_attribute("aria-label", role + "'s message: " + get_message_label_data(message), 0)} class="${[
      "svelte-pcjl1g",
      (i === value.length - 1 ? "latest" : "") + " " + (!render_markdown ? "message-markdown-disabled" : "") + " " + (selectable ? "selectable" : "")
    ].join(" ").trim()}"${add_styles({
      "user-select": `text`,
      "cursor": selectable ? "pointer" : "auto",
      "text-align": rtl ? "right" : "left"
    })}${add_attribute("this", messageElements[thought_index], 0)}>${message?.metadata?.title ? `${validate_component(Thought, "Thought").$$render(
      $$result,
      {
        thought: message,
        rtl,
        sanitize_html,
        latex_delimiters,
        render_markdown,
        _components,
        upload,
        thought_index,
        target,
        root,
        theme_mode,
        _fetch,
        scroll: scroll2,
        allow_file_downloads,
        display_consecutive_in_same_bubble,
        i18n,
        line_breaks
      },
      {},
      {}
    )}` : `${validate_component(MessageContent, "MessageContent").$$render(
      $$result,
      {
        message,
        sanitize_html,
        latex_delimiters,
        render_markdown,
        _components,
        upload,
        thought_index,
        target,
        root,
        theme_mode,
        _fetch,
        scroll: scroll2,
        allow_file_downloads,
        display_consecutive_in_same_bubble,
        i18n,
        line_breaks
      },
      {},
      {}
    )}`} </div>`}</div> ${layout === "panel" ? `${validate_component(ButtonPanel, "ButtonPanel").$$render($$result, Object.assign({}, button_panel_props, { current_feedback }), {}, {})}` : ``}`;
  })}</div></div></div> ${layout === "bubble" ? `${validate_component(ButtonPanel, "ButtonPanel").$$render($$result, Object.assign({}, button_panel_props), {}, {})}` : ``}`;
});
const css$3 = {
  code: ".container.svelte-1u5aj92{display:flex;margin:calc(var(--spacing-xl) * 2)}.bubble.pending.svelte-1u5aj92{border-width:1px;border-radius:var(--radius-lg);border-bottom-left-radius:0;border-color:var(--border-color-primary);background-color:var(--background-fill-secondary);box-shadow:var(--shadow-drop);align-self:flex-start;width:fit-content;margin-bottom:var(--spacing-xl)}.bubble.with_opposite_avatar.svelte-1u5aj92{margin-right:calc(var(--spacing-xxl) + 35px + var(--spacing-xxl))}.panel.pending.svelte-1u5aj92{margin:0;padding:calc(var(--spacing-lg) * 2) calc(var(--spacing-lg) * 2);width:100%;border:none;background:none;box-shadow:none;border-radius:0}.panel.with_avatar.svelte-1u5aj92{padding-left:calc(var(--spacing-xl) * 2) !important;padding-right:calc(var(--spacing-xl) * 2) !important}.avatar-container.svelte-1u5aj92{align-self:flex-start;position:relative;display:flex;justify-content:flex-start;align-items:flex-start;width:35px;height:35px;flex-shrink:0;bottom:0;border-radius:50%;border:1px solid var(--border-color-primary);margin-right:var(--spacing-xxl)}.message-content.svelte-1u5aj92{padding:var(--spacing-sm) var(--spacing-xl);min-height:var(--size-8);display:flex;align-items:center}.dots.svelte-1u5aj92{display:flex;gap:var(--spacing-xs);align-items:center}.dot.svelte-1u5aj92{width:var(--size-1-5);height:var(--size-1-5);margin-right:var(--spacing-xs);border-radius:50%;background-color:var(--body-text-color);opacity:0.5;animation:svelte-1u5aj92-pulse 1.5s infinite}.dot.svelte-1u5aj92:nth-child(2){animation-delay:0.2s}.dot.svelte-1u5aj92:nth-child(3){animation-delay:0.4s}@keyframes svelte-1u5aj92-pulse{0%,100%{opacity:0.4;transform:scale(1)}50%{opacity:1;transform:scale(1.1)}}",
  map: '{"version":3,"file":"Pending.svelte","sources":["Pending.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { Image } from \\"@gradio/image/shared\\";\\nexport let layout = \\"bubble\\";\\nexport let avatar_images = [null, null];\\n<\/script>\\n\\n<div class=\\"container\\">\\n\\t{#if avatar_images[1] !== null}\\n\\t\\t<div class=\\"avatar-container\\">\\n\\t\\t\\t<Image class=\\"avatar-image\\" src={avatar_images[1].url} alt=\\"bot avatar\\" />\\n\\t\\t</div>\\n\\t{/if}\\n\\n\\t<div\\n\\t\\tclass=\\"message bot pending {layout}\\"\\n\\t\\tclass:with_avatar={avatar_images[1] !== null}\\n\\t\\tclass:with_opposite_avatar={avatar_images[0] !== null}\\n\\t\\trole=\\"status\\"\\n\\t\\taria-label=\\"Loading response\\"\\n\\t\\taria-live=\\"polite\\"\\n\\t>\\n\\t\\t<div class=\\"message-content\\">\\n\\t\\t\\t<span class=\\"sr-only\\">Loading content</span>\\n\\t\\t\\t<div class=\\"dots\\">\\n\\t\\t\\t\\t<div class=\\"dot\\" />\\n\\t\\t\\t\\t<div class=\\"dot\\" />\\n\\t\\t\\t\\t<div class=\\"dot\\" />\\n\\t\\t\\t</div>\\n\\t\\t</div>\\n\\t</div>\\n</div>\\n\\n<style>\\n\\t.container {\\n\\t\\tdisplay: flex;\\n\\t\\tmargin: calc(var(--spacing-xl) * 2);\\n\\t}\\n\\n\\t.bubble.pending {\\n\\t\\tborder-width: 1px;\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\tborder-bottom-left-radius: 0;\\n\\t\\tborder-color: var(--border-color-primary);\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t\\talign-self: flex-start;\\n\\t\\twidth: fit-content;\\n\\t\\tmargin-bottom: var(--spacing-xl);\\n\\t}\\n\\n\\t.bubble.with_opposite_avatar {\\n\\t\\tmargin-right: calc(var(--spacing-xxl) + 35px + var(--spacing-xxl));\\n\\t}\\n\\n\\t.panel.pending {\\n\\t\\tmargin: 0;\\n\\t\\tpadding: calc(var(--spacing-lg) * 2) calc(var(--spacing-lg) * 2);\\n\\t\\twidth: 100%;\\n\\t\\tborder: none;\\n\\t\\tbackground: none;\\n\\t\\tbox-shadow: none;\\n\\t\\tborder-radius: 0;\\n\\t}\\n\\n\\t.panel.with_avatar {\\n\\t\\tpadding-left: calc(var(--spacing-xl) * 2) !important;\\n\\t\\tpadding-right: calc(var(--spacing-xl) * 2) !important;\\n\\t}\\n\\n\\t.avatar-container {\\n\\t\\talign-self: flex-start;\\n\\t\\tposition: relative;\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: flex-start;\\n\\t\\talign-items: flex-start;\\n\\t\\twidth: 35px;\\n\\t\\theight: 35px;\\n\\t\\tflex-shrink: 0;\\n\\t\\tbottom: 0;\\n\\t\\tborder-radius: 50%;\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tmargin-right: var(--spacing-xxl);\\n\\t}\\n\\n\\t.message-content {\\n\\t\\tpadding: var(--spacing-sm) var(--spacing-xl);\\n\\t\\tmin-height: var(--size-8);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t}\\n\\n\\t.dots {\\n\\t\\tdisplay: flex;\\n\\t\\tgap: var(--spacing-xs);\\n\\t\\talign-items: center;\\n\\t}\\n\\n\\t.dot {\\n\\t\\twidth: var(--size-1-5);\\n\\t\\theight: var(--size-1-5);\\n\\t\\tmargin-right: var(--spacing-xs);\\n\\t\\tborder-radius: 50%;\\n\\t\\tbackground-color: var(--body-text-color);\\n\\t\\topacity: 0.5;\\n\\t\\tanimation: pulse 1.5s infinite;\\n\\t}\\n\\n\\t.dot:nth-child(2) {\\n\\t\\tanimation-delay: 0.2s;\\n\\t}\\n\\n\\t.dot:nth-child(3) {\\n\\t\\tanimation-delay: 0.4s;\\n\\t}\\n\\n\\t@keyframes pulse {\\n\\t\\t0%,\\n\\t\\t100% {\\n\\t\\t\\topacity: 0.4;\\n\\t\\t\\ttransform: scale(1);\\n\\t\\t}\\n\\t\\t50% {\\n\\t\\t\\topacity: 1;\\n\\t\\t\\ttransform: scale(1.1);\\n\\t\\t}\\n\\t}</style>\\n"],"names":[],"mappings":"AAgCC,yBAAW,CACV,OAAO,CAAE,IAAI,CACb,MAAM,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CACnC,CAEA,OAAO,uBAAS,CACf,YAAY,CAAE,GAAG,CACjB,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,yBAAyB,CAAE,CAAC,CAC5B,YAAY,CAAE,IAAI,sBAAsB,CAAC,CACzC,gBAAgB,CAAE,IAAI,2BAA2B,CAAC,CAClD,UAAU,CAAE,IAAI,aAAa,CAAC,CAC9B,UAAU,CAAE,UAAU,CACtB,KAAK,CAAE,WAAW,CAClB,aAAa,CAAE,IAAI,YAAY,CAChC,CAEA,OAAO,oCAAsB,CAC5B,YAAY,CAAE,KAAK,IAAI,aAAa,CAAC,CAAC,CAAC,CAAC,IAAI,CAAC,CAAC,CAAC,IAAI,aAAa,CAAC,CAClE,CAEA,MAAM,uBAAS,CACd,MAAM,CAAE,CAAC,CACT,OAAO,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,CAChE,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,UAAU,CAAE,IAAI,CAChB,UAAU,CAAE,IAAI,CAChB,aAAa,CAAE,CAChB,CAEA,MAAM,2BAAa,CAClB,YAAY,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,UAAU,CACpD,aAAa,CAAE,KAAK,IAAI,YAAY,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,CAAC,UAC5C,CAEA,gCAAkB,CACjB,UAAU,CAAE,UAAU,CACtB,QAAQ,CAAE,QAAQ,CAClB,OAAO,CAAE,IAAI,CACb,eAAe,CAAE,UAAU,CAC3B,WAAW,CAAE,UAAU,CACvB,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,WAAW,CAAE,CAAC,CACd,MAAM,CAAE,CAAC,CACT,aAAa,CAAE,GAAG,CAClB,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CAAC,CAC7C,YAAY,CAAE,IAAI,aAAa,CAChC,CAEA,+BAAiB,CAChB,OAAO,CAAE,IAAI,YAAY,CAAC,CAAC,IAAI,YAAY,CAAC,CAC5C,UAAU,CAAE,IAAI,QAAQ,CAAC,CACzB,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MACd,CAEA,oBAAM,CACL,OAAO,CAAE,IAAI,CACb,GAAG,CAAE,IAAI,YAAY,CAAC,CACtB,WAAW,CAAE,MACd,CAEA,mBAAK,CACJ,KAAK,CAAE,IAAI,UAAU,CAAC,CACtB,MAAM,CAAE,IAAI,UAAU,CAAC,CACvB,YAAY,CAAE,IAAI,YAAY,CAAC,CAC/B,aAAa,CAAE,GAAG,CAClB,gBAAgB,CAAE,IAAI,iBAAiB,CAAC,CACxC,OAAO,CAAE,GAAG,CACZ,SAAS,CAAE,oBAAK,CAAC,IAAI,CAAC,QACvB,CAEA,mBAAI,WAAW,CAAC,CAAE,CACjB,eAAe,CAAE,IAClB,CAEA,mBAAI,WAAW,CAAC,CAAE,CACjB,eAAe,CAAE,IAClB,CAEA,WAAW,oBAAM,CAChB,EAAE,CACF,IAAK,CACJ,OAAO,CAAE,GAAG,CACZ,SAAS,CAAE,MAAM,CAAC,CACnB,CACA,GAAI,CACH,OAAO,CAAE,CAAC,CACV,SAAS,CAAE,MAAM,GAAG,CACrB,CACD"}'
};
const Pending = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { layout = "bubble" } = $$props;
  let { avatar_images = [null, null] } = $$props;
  if ($$props.layout === void 0 && $$bindings.layout && layout !== void 0)
    $$bindings.layout(layout);
  if ($$props.avatar_images === void 0 && $$bindings.avatar_images && avatar_images !== void 0)
    $$bindings.avatar_images(avatar_images);
  $$result.css.add(css$3);
  return `<div class="container svelte-1u5aj92">${avatar_images[1] !== null ? `<div class="avatar-container svelte-1u5aj92">${validate_component(Image$1, "Image").$$render(
    $$result,
    {
      class: "avatar-image",
      src: avatar_images[1].url,
      alt: "bot avatar"
    },
    {},
    {}
  )}</div>` : ``} <div class="${[
    "message bot pending " + escape(layout, true) + " svelte-1u5aj92",
    (avatar_images[1] !== null ? "with_avatar" : "") + " " + (avatar_images[0] !== null ? "with_opposite_avatar" : "")
  ].join(" ").trim()}" role="status" aria-label="Loading response" aria-live="polite"><div class="message-content svelte-1u5aj92" data-svelte-h="svelte-1vfby8"><span class="sr-only">Loading content</span> <div class="dots svelte-1u5aj92"><div class="dot svelte-1u5aj92"></div> <div class="dot svelte-1u5aj92"></div> <div class="dot svelte-1u5aj92"></div></div></div></div> </div>`;
});
const css$2 = {
  code: ".placeholder-content.svelte-9pi8y1{display:flex;flex-direction:column;height:100%}.placeholder.svelte-9pi8y1{align-items:center;display:flex;justify-content:center;height:100%;flex-grow:1}.examples.svelte-9pi8y1 img{pointer-events:none}.examples.svelte-9pi8y1{margin:auto;padding:var(--spacing-xxl);display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:var(--spacing-xl);max-width:calc(min(4 * 240px + 5 * var(--spacing-xxl), 100%))}.example.svelte-9pi8y1{display:flex;flex-direction:column;align-items:flex-start;padding:var(--spacing-xxl);border:none;border-radius:var(--radius-lg);background-color:var(--block-background-fill);cursor:pointer;transition:all 150ms ease-in-out;width:100%;gap:var(--spacing-sm);border:var(--block-border-width) solid var(--block-border-color);transform:translateY(0px)}.example.svelte-9pi8y1:hover{transform:translateY(-2px);background-color:var(--color-accent-soft)}.example-content.svelte-9pi8y1{display:flex;flex-direction:column;align-items:flex-start;width:100%;height:100%}.example-text-content.svelte-9pi8y1{margin-top:auto;text-align:left}.example-text.svelte-9pi8y1{font-size:var(--text-md);text-align:left;overflow:hidden;text-overflow:ellipsis}.example-icons-grid.svelte-9pi8y1{display:flex;gap:var(--spacing-sm);margin-bottom:var(--spacing-lg);width:100%}.example-icon.svelte-9pi8y1{flex-shrink:0;width:var(--size-8);height:var(--size-8);display:flex;align-items:center;justify-content:center;border-radius:var(--radius-lg);border:var(--block-border-width) solid var(--block-border-color);background-color:var(--block-background-fill);position:relative}.example-icon.svelte-9pi8y1 svg{width:var(--size-4);height:var(--size-4);color:var(--color-text-secondary)}.text-icon-aa.svelte-9pi8y1{font-size:var(--text-sm);font-weight:var(--weight-semibold);color:var(--color-text-secondary);line-height:1}.example-image-container.svelte-9pi8y1{width:var(--size-8);height:var(--size-8);border-radius:var(--radius-lg);overflow:hidden;position:relative;margin-bottom:var(--spacing-lg)}.example-image-container.svelte-9pi8y1 img{width:100%;height:100%;object-fit:cover}.image-overlay.svelte-9pi8y1{position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(0, 0, 0, 0.6);color:white;display:flex;align-items:center;justify-content:center;font-size:var(--text-lg);font-weight:var(--weight-semibold);border-radius:var(--radius-lg)}.file-overlay.svelte-9pi8y1{position:absolute;inset:0;background:rgba(0, 0, 0, 0.6);color:white;display:flex;align-items:center;justify-content:center;font-size:var(--text-sm);font-weight:var(--weight-semibold);border-radius:var(--radius-lg)}",
  map: '{"version":3,"file":"Examples.svelte","sources":["Examples.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { Image } from \\"@gradio/image/shared\\";\\nimport { MarkdownCode as Markdown } from \\"@gradio/markdown-code\\";\\nimport { File, Music, Video } from \\"@gradio/icons\\";\\nimport { createEventDispatcher } from \\"svelte\\";\\nexport let examples = null;\\nexport let placeholder = null;\\nexport let latex_delimiters;\\nexport let root;\\nconst dispatch = createEventDispatcher();\\nfunction handle_example_select(i, example) {\\n    const example_obj = typeof example === \\"string\\" ? { text: example } : example;\\n    dispatch(\\"example_select\\", {\\n        index: i,\\n        value: { text: example_obj.text, files: example_obj.files }\\n    });\\n}\\n<\/script>\\n\\n<div class=\\"placeholder-content\\" role=\\"complementary\\">\\n\\t{#if placeholder !== null}\\n\\t\\t<div class=\\"placeholder\\">\\n\\t\\t\\t<Markdown message={placeholder} {latex_delimiters} {root} />\\n\\t\\t</div>\\n\\t{/if}\\n\\t{#if examples !== null}\\n\\t\\t<div class=\\"examples\\" role=\\"list\\">\\n\\t\\t\\t{#each examples as example, i}\\n\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\tclass=\\"example\\"\\n\\t\\t\\t\\t\\ton:click={() =>\\n\\t\\t\\t\\t\\t\\thandle_example_select(\\n\\t\\t\\t\\t\\t\\t\\ti,\\n\\t\\t\\t\\t\\t\\t\\ttypeof example === \\"string\\" ? { text: example } : example\\n\\t\\t\\t\\t\\t\\t)}\\n\\t\\t\\t\\t\\taria-label={`Select example ${i + 1}: ${example.display_text || example.text}`}\\n\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t<div class=\\"example-content\\">\\n\\t\\t\\t\\t\\t\\t{#if example?.icon?.url}\\n\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-image-container\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t<Image\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-image\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tsrc={example.icon.url}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\talt=\\"Example icon\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t{:else if example?.icon?.mime_type === \\"text\\"}\\n\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-icon\\" aria-hidden=\\"true\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t<span class=\\"text-icon-aa\\">Aa</span>\\n\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t{:else if example.files !== undefined && example.files.length > 0}\\n\\t\\t\\t\\t\\t\\t\\t{#if example.files.length > 1}\\n\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-icons-grid\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\trole=\\"group\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label=\\"Example attachments\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{#each example.files.slice(0, 4) as file, i}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if file.mime_type?.includes(\\"image\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-image-container\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<Image\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-image\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tsrc={file.url}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\talt={file.orig_name || `Example image ${i + 1}`}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if i === 3 && example.files.length > 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"image-overlay\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\trole=\\"status\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`${example.files.length - 4} more files`}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t+{example.files.length - 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{:else if file.mime_type?.includes(\\"video\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-image-container\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<video\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-image\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tsrc={file.url}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-hidden=\\"true\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if i === 3 && example.files.length > 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"image-overlay\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\trole=\\"status\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`${example.files.length - 4} more files`}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t+{example.files.length - 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-icon\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`File: ${file.orig_name}`}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if file.mime_type?.includes(\\"audio\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<Music />\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<File />\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{#if example.files.length > 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-icon\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"file-overlay\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\trole=\\"status\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`${example.files.length - 4} more files`}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t+{example.files.length - 4}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t{:else if example.files[0].mime_type?.includes(\\"image\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-image-container\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<Image\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-image\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tsrc={example.files[0].url}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\talt={example.files[0].orig_name || \\"Example image\\"}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t{:else if example.files[0].mime_type?.includes(\\"video\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t<div class=\\"example-image-container\\">\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<video\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-image\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tsrc={example.files[0].url}\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\taria-hidden=\\"true\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t{:else if example.files[0].mime_type?.includes(\\"audio\\")}\\n\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-icon\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`File: ${example.files[0].orig_name}`}\\n\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<Music />\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"example-icon\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t\\taria-label={`File: ${example.files[0].orig_name}`}\\n\\t\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t<File />\\n\\t\\t\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\n\\t\\t\\t\\t\\t\\t<div class=\\"example-text-content\\">\\n\\t\\t\\t\\t\\t\\t\\t<span class=\\"example-text\\"\\n\\t\\t\\t\\t\\t\\t\\t\\t>{example.display_text || example.text}</span\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t\\t</div>\\n\\t\\t\\t\\t</button>\\n\\t\\t\\t{/each}\\n\\t\\t</div>\\n\\t{/if}\\n</div>\\n\\n<style>\\n\\t.placeholder-content {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\theight: 100%;\\n\\t}\\n\\n\\t.placeholder {\\n\\t\\talign-items: center;\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: center;\\n\\t\\theight: 100%;\\n\\t\\tflex-grow: 1;\\n\\t}\\n\\n\\t.examples :global(img) {\\n\\t\\tpointer-events: none;\\n\\t}\\n\\n\\t.examples {\\n\\t\\tmargin: auto;\\n\\t\\tpadding: var(--spacing-xxl);\\n\\t\\tdisplay: grid;\\n\\t\\tgrid-template-columns: repeat(auto-fit, minmax(240px, 1fr));\\n\\t\\tgap: var(--spacing-xl);\\n\\t\\tmax-width: calc(min(4 * 240px + 5 * var(--spacing-xxl), 100%));\\n\\t}\\n\\n\\t.example {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\talign-items: flex-start;\\n\\t\\tpadding: var(--spacing-xxl);\\n\\t\\tborder: none;\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\tbackground-color: var(--block-background-fill);\\n\\t\\tcursor: pointer;\\n\\t\\ttransition: all 150ms ease-in-out;\\n\\t\\twidth: 100%;\\n\\t\\tgap: var(--spacing-sm);\\n\\t\\tborder: var(--block-border-width) solid var(--block-border-color);\\n\\t\\ttransform: translateY(0px);\\n\\t}\\n\\n\\t.example:hover {\\n\\t\\ttransform: translateY(-2px);\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t}\\n\\n\\t.example-content {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\talign-items: flex-start;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t}\\n\\n\\t.example-text-content {\\n\\t\\tmargin-top: auto;\\n\\t\\ttext-align: left;\\n\\t}\\n\\n\\t.example-text {\\n\\t\\tfont-size: var(--text-md);\\n\\t\\ttext-align: left;\\n\\t\\toverflow: hidden;\\n\\t\\ttext-overflow: ellipsis;\\n\\t}\\n\\n\\t.example-icons-grid {\\n\\t\\tdisplay: flex;\\n\\t\\tgap: var(--spacing-sm);\\n\\t\\tmargin-bottom: var(--spacing-lg);\\n\\t\\twidth: 100%;\\n\\t}\\n\\n\\t.example-icon {\\n\\t\\tflex-shrink: 0;\\n\\t\\twidth: var(--size-8);\\n\\t\\theight: var(--size-8);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\tborder: var(--block-border-width) solid var(--block-border-color);\\n\\t\\tbackground-color: var(--block-background-fill);\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.example-icon :global(svg) {\\n\\t\\twidth: var(--size-4);\\n\\t\\theight: var(--size-4);\\n\\t\\tcolor: var(--color-text-secondary);\\n\\t}\\n\\n\\t.text-icon-aa {\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tfont-weight: var(--weight-semibold);\\n\\t\\tcolor: var(--color-text-secondary);\\n\\t\\tline-height: 1;\\n\\t}\\n\\n\\t.example-image-container {\\n\\t\\twidth: var(--size-8);\\n\\t\\theight: var(--size-8);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\toverflow: hidden;\\n\\t\\tposition: relative;\\n\\t\\tmargin-bottom: var(--spacing-lg);\\n\\t}\\n\\n\\t.example-image-container :global(img) {\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\tobject-fit: cover;\\n\\t}\\n\\n\\t.image-overlay {\\n\\t\\tposition: absolute;\\n\\t\\ttop: 0;\\n\\t\\tleft: 0;\\n\\t\\tright: 0;\\n\\t\\tbottom: 0;\\n\\t\\tbackground: rgba(0, 0, 0, 0.6);\\n\\t\\tcolor: white;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tfont-size: var(--text-lg);\\n\\t\\tfont-weight: var(--weight-semibold);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t}\\n\\n\\t.file-overlay {\\n\\t\\tposition: absolute;\\n\\t\\tinset: 0;\\n\\t\\tbackground: rgba(0, 0, 0, 0.6);\\n\\t\\tcolor: white;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tfont-weight: var(--weight-semibold);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t}</style>\\n"],"names":[],"mappings":"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"}'
};
const Examples = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { examples = null } = $$props;
  let { placeholder = null } = $$props;
  let { latex_delimiters } = $$props;
  let { root } = $$props;
  createEventDispatcher();
  if ($$props.examples === void 0 && $$bindings.examples && examples !== void 0)
    $$bindings.examples(examples);
  if ($$props.placeholder === void 0 && $$bindings.placeholder && placeholder !== void 0)
    $$bindings.placeholder(placeholder);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  $$result.css.add(css$2);
  return `<div class="placeholder-content svelte-9pi8y1" role="complementary">${placeholder !== null ? `<div class="placeholder svelte-9pi8y1">${validate_component(MarkdownCode, "Markdown").$$render(
    $$result,
    {
      message: placeholder,
      latex_delimiters,
      root
    },
    {},
    {}
  )}</div>` : ``} ${examples !== null ? `<div class="examples svelte-9pi8y1" role="list">${each(examples, (example, i) => {
    return `<button class="example svelte-9pi8y1"${add_attribute("aria-label", `Select example ${i + 1}: ${example.display_text || example.text}`, 0)}><div class="example-content svelte-9pi8y1">${example?.icon?.url ? `<div class="example-image-container svelte-9pi8y1">${validate_component(Image$1, "Image").$$render(
      $$result,
      {
        class: "example-image",
        src: example.icon.url,
        alt: "Example icon"
      },
      {},
      {}
    )} </div>` : `${example?.icon?.mime_type === "text" ? `<div class="example-icon svelte-9pi8y1" aria-hidden="true" data-svelte-h="svelte-15cq9iz"><span class="text-icon-aa svelte-9pi8y1">Aa</span> </div>` : `${example.files !== void 0 && example.files.length > 0 ? `${example.files.length > 1 ? `<div class="example-icons-grid svelte-9pi8y1" role="group" aria-label="Example attachments">${each(example.files.slice(0, 4), (file, i2) => {
      return `${file.mime_type?.includes("image") ? `<div class="example-image-container svelte-9pi8y1">${validate_component(Image$1, "Image").$$render(
        $$result,
        {
          class: "example-image",
          src: file.url,
          alt: file.orig_name || `Example image ${i2 + 1}`
        },
        {},
        {}
      )} ${i2 === 3 && example.files.length > 4 ? `<div class="image-overlay svelte-9pi8y1" role="status"${add_attribute("aria-label", `${example.files.length - 4} more files`, 0)}>+${escape(example.files.length - 4)} </div>` : ``} </div>` : `${file.mime_type?.includes("video") ? `<div class="example-image-container svelte-9pi8y1"><video class="example-image"${add_attribute("src", file.url, 0)} aria-hidden="true"></video> ${i2 === 3 && example.files.length > 4 ? `<div class="image-overlay svelte-9pi8y1" role="status"${add_attribute("aria-label", `${example.files.length - 4} more files`, 0)}>+${escape(example.files.length - 4)} </div>` : ``} </div>` : `<div class="example-icon svelte-9pi8y1"${add_attribute("aria-label", `File: ${file.orig_name}`, 0)}>${file.mime_type?.includes("audio") ? `${validate_component(Music, "Music").$$render($$result, {}, {}, {})}` : `${validate_component(File$1, "File").$$render($$result, {}, {}, {})}`} </div>`}`}`;
    })} ${example.files.length > 4 ? `<div class="example-icon svelte-9pi8y1"><div class="file-overlay svelte-9pi8y1" role="status"${add_attribute("aria-label", `${example.files.length - 4} more files`, 0)}>+${escape(example.files.length - 4)}</div> </div>` : ``} </div>` : `${example.files[0].mime_type?.includes("image") ? `<div class="example-image-container svelte-9pi8y1">${validate_component(Image$1, "Image").$$render(
      $$result,
      {
        class: "example-image",
        src: example.files[0].url,
        alt: example.files[0].orig_name || "Example image"
      },
      {},
      {}
    )} </div>` : `${example.files[0].mime_type?.includes("video") ? `<div class="example-image-container svelte-9pi8y1"><video class="example-image"${add_attribute("src", example.files[0].url, 0)} aria-hidden="true"></video> </div>` : `${example.files[0].mime_type?.includes("audio") ? `<div class="example-icon svelte-9pi8y1"${add_attribute("aria-label", `File: ${example.files[0].orig_name}`, 0)}>${validate_component(Music, "Music").$$render($$result, {}, {}, {})} </div>` : `<div class="example-icon svelte-9pi8y1"${add_attribute("aria-label", `File: ${example.files[0].orig_name}`, 0)}>${validate_component(File$1, "File").$$render($$result, {}, {}, {})} </div>`}`}`}`}` : ``}`}`} <div class="example-text-content svelte-9pi8y1"><span class="example-text svelte-9pi8y1">${escape(example.display_text || example.text)}</span> </div></div> </button>`;
  })}</div>` : ``} </div>`;
});
const CopyAll = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  onDestroy(() => {
  });
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  return `${validate_component(IconButton, "IconButton").$$render(
    $$result,
    {
      Icon: Copy,
      label: "Copy conversation"
    },
    {},
    {}
  )}`;
});
const css$1 = {
  code: ".panel-wrap.svelte-gjtrl6.svelte-gjtrl6{width:100%;overflow-y:auto}.bubble-wrap.svelte-gjtrl6.svelte-gjtrl6{width:100%;overflow-y:auto;height:100%;padding-top:var(--spacing-xxl)}@media(prefers-color-scheme: dark){.bubble-wrap.svelte-gjtrl6.svelte-gjtrl6{background:var(--background-fill-secondary)}}.message-wrap.svelte-gjtrl6 .prose.chatbot.md{opacity:0.8;overflow-wrap:break-word}.message-wrap.svelte-gjtrl6 .message-row .md img{border-radius:var(--radius-xl);margin:var(--size-2);width:400px;max-width:30vw;max-height:30vw}.message-wrap.svelte-gjtrl6 .message a{color:var(--color-text-link);text-decoration:underline}.message-wrap.svelte-gjtrl6 .bot:not(:has(.table-wrap)) table,.message-wrap.svelte-gjtrl6 .bot:not(:has(.table-wrap)) tr,.message-wrap.svelte-gjtrl6 .bot:not(:has(.table-wrap)) td,.message-wrap.svelte-gjtrl6 .bot:not(:has(.table-wrap)) th{border:1px solid var(--border-color-primary)}.message-wrap.svelte-gjtrl6 .user table,.message-wrap.svelte-gjtrl6 .user tr,.message-wrap.svelte-gjtrl6 .user td,.message-wrap.svelte-gjtrl6 .user th{border:1px solid var(--border-color-accent)}.message-wrap.svelte-gjtrl6 span.katex{font-size:var(--text-lg);direction:ltr}.message-wrap.svelte-gjtrl6 span.katex-display{margin-top:0}.message-wrap.svelte-gjtrl6 pre{position:relative}.message-wrap.svelte-gjtrl6 .grid-wrap{max-height:80% !important;max-width:600px;object-fit:contain}.message-wrap.svelte-gjtrl6>div.svelte-gjtrl6 p:not(:first-child){margin-top:var(--spacing-xxl)}.message-wrap.svelte-gjtrl6.svelte-gjtrl6{display:flex;flex-direction:column;justify-content:space-between;margin-bottom:var(--spacing-xxl)}.panel-wrap.svelte-gjtrl6 .message-row:first-child{padding-top:calc(var(--spacing-xxl) * 2)}.scroll-down-button-container.svelte-gjtrl6.svelte-gjtrl6{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);z-index:var(--layer-top)}.scroll-down-button-container.svelte-gjtrl6 button{border-radius:50%;box-shadow:var(--shadow-drop);transition:box-shadow 0.2s ease-in-out,\n			transform 0.2s ease-in-out}.scroll-down-button-container.svelte-gjtrl6 button:hover{box-shadow:var(--shadow-drop),\n			0 2px 2px rgba(0, 0, 0, 0.05);transform:translateY(-2px)}.options.svelte-gjtrl6.svelte-gjtrl6{margin-left:auto;padding:var(--spacing-xxl);display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:var(--spacing-xxl);max-width:calc(min(4 * 200px + 5 * var(--spacing-xxl), 100%));justify-content:end}.option.svelte-gjtrl6.svelte-gjtrl6{display:flex;flex-direction:column;align-items:center;padding:var(--spacing-xl);border:1px dashed var(--border-color-primary);border-radius:var(--radius-md);background-color:var(--background-fill-secondary);cursor:pointer;transition:var(--button-transition);max-width:var(--size-56);width:100%;justify-content:center}.option.svelte-gjtrl6.svelte-gjtrl6:hover{background-color:var(--color-accent-soft);border-color:var(--border-color-accent)}",
  map: '{"version":3,"file":"ChatBot.svelte","sources":["ChatBot.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { format_chat_for_sharing, is_last_bot_message, group_messages, load_components, get_components_from_messages } from \\"./utils\\";\\nimport { copy } from \\"@gradio/utils\\";\\nimport Message from \\"./Message.svelte\\";\\nimport { dequal } from \\"dequal/lite\\";\\nimport { createEventDispatcher, tick, onMount } from \\"svelte\\";\\nimport { Trash, Community, ScrollDownArrow } from \\"@gradio/icons\\";\\nimport { IconButtonWrapper, IconButton } from \\"@gradio/atoms\\";\\nimport Pending from \\"./Pending.svelte\\";\\nimport { ShareError } from \\"@gradio/utils\\";\\nimport { Gradio } from \\"@gradio/utils\\";\\nimport Examples from \\"./Examples.svelte\\";\\nexport let value = [];\\nlet old_value = null;\\nimport CopyAll from \\"./CopyAll.svelte\\";\\nexport let _fetch;\\nexport let load_component;\\nexport let allow_file_downloads;\\nexport let display_consecutive_in_same_bubble;\\nlet _components = {};\\nconst is_browser = typeof window !== \\"undefined\\";\\nasync function update_components() {\\n    _components = await load_components(get_components_from_messages(value), _components, load_component);\\n}\\n$: value, update_components();\\nexport let latex_delimiters;\\nexport let pending_message = false;\\nexport let generating = false;\\nexport let selectable = false;\\nexport let likeable = false;\\nexport let feedback_options;\\nexport let feedback_value = null;\\nexport let editable = null;\\nexport let show_share_button = false;\\nexport let show_copy_all_button = false;\\nexport let rtl = false;\\nexport let show_copy_button = false;\\nexport let avatar_images = [null, null];\\nexport let sanitize_html = true;\\nexport let render_markdown = true;\\nexport let line_breaks = true;\\nexport let autoscroll = true;\\nexport let theme_mode;\\nexport let i18n;\\nexport let layout = \\"bubble\\";\\nexport let placeholder = null;\\nexport let upload;\\nexport let msg_format = \\"tuples\\";\\nexport let examples = null;\\nexport let _retryable = false;\\nexport let _undoable = false;\\nexport let like_user_message = false;\\nexport let root;\\nlet target = null;\\nlet edit_index = null;\\nlet edit_message = \\"\\";\\nonMount(() => {\\n    target = document.querySelector(\\"div.gradio-container\\");\\n});\\nlet div;\\nlet show_scroll_button = false;\\nconst dispatch = createEventDispatcher();\\nfunction is_at_bottom() {\\n    return div && div.offsetHeight + div.scrollTop > div.scrollHeight - 100;\\n}\\nfunction scroll_to_bottom() {\\n    if (!div)\\n        return;\\n    div.scrollTo(0, div.scrollHeight);\\n    show_scroll_button = false;\\n}\\nlet scroll_after_component_load = false;\\nasync function scroll_on_value_update() {\\n    if (!autoscroll)\\n        return;\\n    if (is_at_bottom()) {\\n        scroll_after_component_load = true;\\n        await tick();\\n        scroll_to_bottom();\\n    }\\n    else {\\n        show_scroll_button = true;\\n    }\\n}\\nonMount(() => {\\n    scroll_on_value_update();\\n});\\n$: if (value || pending_message || _components) {\\n    scroll_on_value_update();\\n}\\nonMount(() => {\\n    function handle_scroll() {\\n        if (is_at_bottom()) {\\n            show_scroll_button = false;\\n        }\\n        else {\\n            scroll_after_component_load = false;\\n        }\\n    }\\n    div?.addEventListener(\\"scroll\\", handle_scroll);\\n    return () => {\\n        div?.removeEventListener(\\"scroll\\", handle_scroll);\\n    };\\n});\\n$: {\\n    if (!dequal(value, old_value)) {\\n        old_value = value;\\n        dispatch(\\"change\\");\\n    }\\n}\\n$: groupedMessages = value && group_messages(value, msg_format);\\n$: options = value && get_last_bot_options();\\nfunction handle_action(i, message, selected) {\\n    if (selected === \\"undo\\" || selected === \\"retry\\") {\\n        const val_ = value;\\n        let last_index = val_.length - 1;\\n        while (val_[last_index].role === \\"assistant\\") {\\n            last_index--;\\n        }\\n        dispatch(selected, {\\n            index: val_[last_index].index,\\n            value: val_[last_index].content\\n        });\\n    }\\n    else if (selected == \\"edit\\") {\\n        edit_index = i;\\n        edit_message = message.content;\\n    }\\n    else if (selected == \\"edit_cancel\\") {\\n        edit_index = null;\\n    }\\n    else if (selected == \\"edit_submit\\") {\\n        edit_index = null;\\n        dispatch(\\"edit\\", {\\n            index: message.index,\\n            value: edit_message,\\n            previous_value: message.content\\n        });\\n    }\\n    else {\\n        let feedback = selected === \\"Like\\" ? true : selected === \\"Dislike\\" ? false : selected || \\"\\";\\n        if (msg_format === \\"tuples\\") {\\n            dispatch(\\"like\\", {\\n                index: message.index,\\n                value: message.content,\\n                liked: feedback\\n            });\\n        }\\n        else {\\n            if (!groupedMessages)\\n                return;\\n            const message_group = groupedMessages[i];\\n            const [first, last] = [\\n                message_group[0],\\n                message_group[message_group.length - 1]\\n            ];\\n            dispatch(\\"like\\", {\\n                index: first.index,\\n                value: message_group.map((m) => m.content),\\n                liked: feedback\\n            });\\n        }\\n    }\\n}\\nfunction get_last_bot_options() {\\n    if (!value || !groupedMessages || groupedMessages.length === 0)\\n        return void 0;\\n    const last_group = groupedMessages[groupedMessages.length - 1];\\n    if (last_group[0].role !== \\"assistant\\")\\n        return void 0;\\n    return last_group[last_group.length - 1].options;\\n}\\n<\/script>\\n\\n{#if value !== null && value.length > 0}\\n\\t<IconButtonWrapper>\\n\\t\\t{#if show_share_button}\\n\\t\\t\\t<IconButton\\n\\t\\t\\t\\tIcon={Community}\\n\\t\\t\\t\\ton:click={async () => {\\n\\t\\t\\t\\t\\ttry {\\n\\t\\t\\t\\t\\t\\t// @ts-ignore\\n\\t\\t\\t\\t\\t\\tconst formatted = await format_chat_for_sharing(value);\\n\\t\\t\\t\\t\\t\\tdispatch(\\"share\\", {\\n\\t\\t\\t\\t\\t\\t\\tdescription: formatted\\n\\t\\t\\t\\t\\t\\t});\\n\\t\\t\\t\\t\\t} catch (e) {\\n\\t\\t\\t\\t\\t\\tconsole.error(e);\\n\\t\\t\\t\\t\\t\\tlet message = e instanceof ShareError ? e.message : \\"Share failed.\\";\\n\\t\\t\\t\\t\\t\\tdispatch(\\"error\\", message);\\n\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t}}\\n\\t\\t\\t/>\\n\\t\\t{/if}\\n\\t\\t<IconButton Icon={Trash} on:click={() => dispatch(\\"clear\\")} label={\\"Clear\\"}\\n\\t\\t></IconButton>\\n\\t\\t{#if show_copy_all_button}\\n\\t\\t\\t<CopyAll {value} />\\n\\t\\t{/if}\\n\\t</IconButtonWrapper>\\n{/if}\\n\\n<div\\n\\tclass={layout === \\"bubble\\" ? \\"bubble-wrap\\" : \\"panel-wrap\\"}\\n\\tbind:this={div}\\n\\trole=\\"log\\"\\n\\taria-label=\\"chatbot conversation\\"\\n\\taria-live=\\"polite\\"\\n>\\n\\t{#if value !== null && value.length > 0 && groupedMessages !== null}\\n\\t\\t<div class=\\"message-wrap\\" use:copy>\\n\\t\\t\\t{#each groupedMessages as messages, i}\\n\\t\\t\\t\\t{@const role = messages[0].role === \\"user\\" ? \\"user\\" : \\"bot\\"}\\n\\t\\t\\t\\t{@const avatar_img = avatar_images[role === \\"user\\" ? 0 : 1]}\\n\\t\\t\\t\\t{@const opposite_avatar_img = avatar_images[role === \\"user\\" ? 0 : 1]}\\n\\t\\t\\t\\t{@const feedback_index = groupedMessages\\n\\t\\t\\t\\t\\t.slice(0, i)\\n\\t\\t\\t\\t\\t.filter((m) => m[0].role === \\"assistant\\").length}\\n\\t\\t\\t\\t{@const current_feedback =\\n\\t\\t\\t\\t\\trole === \\"bot\\" && feedback_value && feedback_value[feedback_index]\\n\\t\\t\\t\\t\\t\\t? feedback_value[feedback_index]\\n\\t\\t\\t\\t\\t\\t: null}\\n\\t\\t\\t\\t<Message\\n\\t\\t\\t\\t\\t{messages}\\n\\t\\t\\t\\t\\t{display_consecutive_in_same_bubble}\\n\\t\\t\\t\\t\\t{opposite_avatar_img}\\n\\t\\t\\t\\t\\t{avatar_img}\\n\\t\\t\\t\\t\\t{role}\\n\\t\\t\\t\\t\\t{layout}\\n\\t\\t\\t\\t\\t{dispatch}\\n\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\t{_fetch}\\n\\t\\t\\t\\t\\t{line_breaks}\\n\\t\\t\\t\\t\\t{theme_mode}\\n\\t\\t\\t\\t\\t{target}\\n\\t\\t\\t\\t\\t{root}\\n\\t\\t\\t\\t\\t{upload}\\n\\t\\t\\t\\t\\t{selectable}\\n\\t\\t\\t\\t\\t{sanitize_html}\\n\\t\\t\\t\\t\\t{render_markdown}\\n\\t\\t\\t\\t\\t{rtl}\\n\\t\\t\\t\\t\\t{i}\\n\\t\\t\\t\\t\\t{value}\\n\\t\\t\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t\\t\\t{_components}\\n\\t\\t\\t\\t\\t{generating}\\n\\t\\t\\t\\t\\t{msg_format}\\n\\t\\t\\t\\t\\t{feedback_options}\\n\\t\\t\\t\\t\\t{current_feedback}\\n\\t\\t\\t\\t\\tshow_like={role === \\"user\\" ? likeable && like_user_message : likeable}\\n\\t\\t\\t\\t\\tshow_retry={_retryable && is_last_bot_message(messages, value)}\\n\\t\\t\\t\\t\\tshow_undo={_undoable && is_last_bot_message(messages, value)}\\n\\t\\t\\t\\t\\tshow_edit={editable === \\"all\\" ||\\n\\t\\t\\t\\t\\t\\t(editable == \\"user\\" &&\\n\\t\\t\\t\\t\\t\\t\\trole === \\"user\\" &&\\n\\t\\t\\t\\t\\t\\t\\tmessages.length > 0 &&\\n\\t\\t\\t\\t\\t\\t\\tmessages[messages.length - 1].type == \\"text\\")}\\n\\t\\t\\t\\t\\tin_edit_mode={edit_index === i}\\n\\t\\t\\t\\t\\tbind:edit_message\\n\\t\\t\\t\\t\\t{show_copy_button}\\n\\t\\t\\t\\t\\thandle_action={(selected) => handle_action(i, messages[0], selected)}\\n\\t\\t\\t\\t\\tscroll={is_browser ? scroll : () => {}}\\n\\t\\t\\t\\t\\t{allow_file_downloads}\\n\\t\\t\\t\\t\\ton:copy={(e) => dispatch(\\"copy\\", e.detail)}\\n\\t\\t\\t\\t/>\\n\\t\\t\\t{/each}\\n\\t\\t\\t{#if pending_message}\\n\\t\\t\\t\\t<Pending {layout} {avatar_images} />\\n\\t\\t\\t{:else if options}\\n\\t\\t\\t\\t<div class=\\"options\\">\\n\\t\\t\\t\\t\\t{#each options as option, index}\\n\\t\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\t\\tclass=\\"option\\"\\n\\t\\t\\t\\t\\t\\t\\ton:click={() =>\\n\\t\\t\\t\\t\\t\\t\\t\\tdispatch(\\"option_select\\", {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tindex: index,\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tvalue: option.value\\n\\t\\t\\t\\t\\t\\t\\t\\t})}\\n\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t{option.label || option.value}\\n\\t\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t</div>\\n\\t\\t\\t{/if}\\n\\t\\t</div>\\n\\t{:else}\\n\\t\\t<Examples\\n\\t\\t\\t{examples}\\n\\t\\t\\t{placeholder}\\n\\t\\t\\t{latex_delimiters}\\n\\t\\t\\t{root}\\n\\t\\t\\ton:example_select={(e) => dispatch(\\"example_select\\", e.detail)}\\n\\t\\t/>\\n\\t{/if}\\n</div>\\n\\n{#if show_scroll_button}\\n\\t<div class=\\"scroll-down-button-container\\">\\n\\t\\t<IconButton\\n\\t\\t\\tIcon={ScrollDownArrow}\\n\\t\\t\\tlabel=\\"Scroll down\\"\\n\\t\\t\\tsize=\\"large\\"\\n\\t\\t\\ton:click={scroll_to_bottom}\\n\\t\\t/>\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\t.panel-wrap {\\n\\t\\twidth: 100%;\\n\\t\\toverflow-y: auto;\\n\\t}\\n\\n\\t.bubble-wrap {\\n\\t\\twidth: 100%;\\n\\t\\toverflow-y: auto;\\n\\t\\theight: 100%;\\n\\t\\tpadding-top: var(--spacing-xxl);\\n\\t}\\n\\n\\t@media (prefers-color-scheme: dark) {\\n\\t\\t.bubble-wrap {\\n\\t\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\t}\\n\\t}\\n\\n\\t.message-wrap :global(.prose.chatbot.md) {\\n\\t\\topacity: 0.8;\\n\\t\\toverflow-wrap: break-word;\\n\\t}\\n\\n\\t.message-wrap :global(.message-row .md img) {\\n\\t\\tborder-radius: var(--radius-xl);\\n\\t\\tmargin: var(--size-2);\\n\\t\\twidth: 400px;\\n\\t\\tmax-width: 30vw;\\n\\t\\tmax-height: 30vw;\\n\\t}\\n\\n\\t/* link styles */\\n\\t.message-wrap :global(.message a) {\\n\\t\\tcolor: var(--color-text-link);\\n\\t\\ttext-decoration: underline;\\n\\t}\\n\\n\\t/* table styles */\\n\\t.message-wrap :global(.bot:not(:has(.table-wrap)) table),\\n\\t.message-wrap :global(.bot:not(:has(.table-wrap)) tr),\\n\\t.message-wrap :global(.bot:not(:has(.table-wrap)) td),\\n\\t.message-wrap :global(.bot:not(:has(.table-wrap)) th) {\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t}\\n\\n\\t.message-wrap :global(.user table),\\n\\t.message-wrap :global(.user tr),\\n\\t.message-wrap :global(.user td),\\n\\t.message-wrap :global(.user th) {\\n\\t\\tborder: 1px solid var(--border-color-accent);\\n\\t}\\n\\n\\t/* KaTeX */\\n\\t.message-wrap :global(span.katex) {\\n\\t\\tfont-size: var(--text-lg);\\n\\t\\tdirection: ltr;\\n\\t}\\n\\n\\t.message-wrap :global(span.katex-display) {\\n\\t\\tmargin-top: 0;\\n\\t}\\n\\n\\t.message-wrap :global(pre) {\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.message-wrap :global(.grid-wrap) {\\n\\t\\tmax-height: 80% !important;\\n\\t\\tmax-width: 600px;\\n\\t\\tobject-fit: contain;\\n\\t}\\n\\n\\t.message-wrap > div :global(p:not(:first-child)) {\\n\\t\\tmargin-top: var(--spacing-xxl);\\n\\t}\\n\\n\\t.message-wrap {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tjustify-content: space-between;\\n\\t\\tmargin-bottom: var(--spacing-xxl);\\n\\t}\\n\\n\\t.panel-wrap :global(.message-row:first-child) {\\n\\t\\tpadding-top: calc(var(--spacing-xxl) * 2);\\n\\t}\\n\\n\\t.scroll-down-button-container {\\n\\t\\tposition: absolute;\\n\\t\\tbottom: 10px;\\n\\t\\tleft: 50%;\\n\\t\\ttransform: translateX(-50%);\\n\\t\\tz-index: var(--layer-top);\\n\\t}\\n\\t.scroll-down-button-container :global(button) {\\n\\t\\tborder-radius: 50%;\\n\\t\\tbox-shadow: var(--shadow-drop);\\n\\t\\ttransition:\\n\\t\\t\\tbox-shadow 0.2s ease-in-out,\\n\\t\\t\\ttransform 0.2s ease-in-out;\\n\\t}\\n\\t.scroll-down-button-container :global(button:hover) {\\n\\t\\tbox-shadow:\\n\\t\\t\\tvar(--shadow-drop),\\n\\t\\t\\t0 2px 2px rgba(0, 0, 0, 0.05);\\n\\t\\ttransform: translateY(-2px);\\n\\t}\\n\\n\\t.options {\\n\\t\\tmargin-left: auto;\\n\\t\\tpadding: var(--spacing-xxl);\\n\\t\\tdisplay: grid;\\n\\t\\tgrid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\\n\\t\\tgap: var(--spacing-xxl);\\n\\t\\tmax-width: calc(min(4 * 200px + 5 * var(--spacing-xxl), 100%));\\n\\t\\tjustify-content: end;\\n\\t}\\n\\n\\t.option {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\talign-items: center;\\n\\t\\tpadding: var(--spacing-xl);\\n\\t\\tborder: 1px dashed var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t\\tcursor: pointer;\\n\\t\\ttransition: var(--button-transition);\\n\\t\\tmax-width: var(--size-56);\\n\\t\\twidth: 100%;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.option:hover {\\n\\t\\tbackground-color: var(--color-accent-soft);\\n\\t\\tborder-color: var(--border-color-accent);\\n\\t}</style>\\n"],"names":[],"mappings":"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"}'
};
const ChatBot = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let groupedMessages;
  let options;
  let { value = [] } = $$props;
  let old_value = null;
  let { _fetch } = $$props;
  let { load_component } = $$props;
  let { allow_file_downloads } = $$props;
  let { display_consecutive_in_same_bubble } = $$props;
  let _components = {};
  const is_browser = typeof window !== "undefined";
  async function update_components() {
    _components = await load_components(get_components_from_messages(value), _components, load_component);
  }
  let { latex_delimiters } = $$props;
  let { pending_message = false } = $$props;
  let { generating = false } = $$props;
  let { selectable = false } = $$props;
  let { likeable = false } = $$props;
  let { feedback_options } = $$props;
  let { feedback_value = null } = $$props;
  let { editable = null } = $$props;
  let { show_share_button = false } = $$props;
  let { show_copy_all_button = false } = $$props;
  let { rtl = false } = $$props;
  let { show_copy_button = false } = $$props;
  let { avatar_images = [null, null] } = $$props;
  let { sanitize_html = true } = $$props;
  let { render_markdown = true } = $$props;
  let { line_breaks = true } = $$props;
  let { autoscroll = true } = $$props;
  let { theme_mode } = $$props;
  let { i18n } = $$props;
  let { layout = "bubble" } = $$props;
  let { placeholder = null } = $$props;
  let { upload } = $$props;
  let { msg_format = "tuples" } = $$props;
  let { examples = null } = $$props;
  let { _retryable = false } = $$props;
  let { _undoable = false } = $$props;
  let { like_user_message = false } = $$props;
  let { root } = $$props;
  let target = null;
  let edit_index = null;
  let edit_message = "";
  let div;
  let show_scroll_button = false;
  const dispatch = createEventDispatcher();
  async function scroll_on_value_update() {
    if (!autoscroll)
      return;
    {
      show_scroll_button = true;
    }
  }
  function handle_action(i, message, selected) {
    if (selected === "undo" || selected === "retry") {
      const val_ = value;
      let last_index = val_.length - 1;
      while (val_[last_index].role === "assistant") {
        last_index--;
      }
      dispatch(selected, {
        index: val_[last_index].index,
        value: val_[last_index].content
      });
    } else if (selected == "edit") {
      edit_index = i;
      edit_message = message.content;
    } else if (selected == "edit_cancel") {
      edit_index = null;
    } else if (selected == "edit_submit") {
      edit_index = null;
      dispatch("edit", {
        index: message.index,
        value: edit_message,
        previous_value: message.content
      });
    } else {
      let feedback = selected === "Like" ? true : selected === "Dislike" ? false : selected || "";
      if (msg_format === "tuples") {
        dispatch("like", {
          index: message.index,
          value: message.content,
          liked: feedback
        });
      } else {
        if (!groupedMessages)
          return;
        const message_group = groupedMessages[i];
        const [first, last] = [message_group[0], message_group[message_group.length - 1]];
        dispatch("like", {
          index: first.index,
          value: message_group.map((m) => m.content),
          liked: feedback
        });
      }
    }
  }
  function get_last_bot_options() {
    if (!value || !groupedMessages || groupedMessages.length === 0)
      return void 0;
    const last_group = groupedMessages[groupedMessages.length - 1];
    if (last_group[0].role !== "assistant")
      return void 0;
    return last_group[last_group.length - 1].options;
  }
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props._fetch === void 0 && $$bindings._fetch && _fetch !== void 0)
    $$bindings._fetch(_fetch);
  if ($$props.load_component === void 0 && $$bindings.load_component && load_component !== void 0)
    $$bindings.load_component(load_component);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  if ($$props.display_consecutive_in_same_bubble === void 0 && $$bindings.display_consecutive_in_same_bubble && display_consecutive_in_same_bubble !== void 0)
    $$bindings.display_consecutive_in_same_bubble(display_consecutive_in_same_bubble);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.pending_message === void 0 && $$bindings.pending_message && pending_message !== void 0)
    $$bindings.pending_message(pending_message);
  if ($$props.generating === void 0 && $$bindings.generating && generating !== void 0)
    $$bindings.generating(generating);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.likeable === void 0 && $$bindings.likeable && likeable !== void 0)
    $$bindings.likeable(likeable);
  if ($$props.feedback_options === void 0 && $$bindings.feedback_options && feedback_options !== void 0)
    $$bindings.feedback_options(feedback_options);
  if ($$props.feedback_value === void 0 && $$bindings.feedback_value && feedback_value !== void 0)
    $$bindings.feedback_value(feedback_value);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.show_share_button === void 0 && $$bindings.show_share_button && show_share_button !== void 0)
    $$bindings.show_share_button(show_share_button);
  if ($$props.show_copy_all_button === void 0 && $$bindings.show_copy_all_button && show_copy_all_button !== void 0)
    $$bindings.show_copy_all_button(show_copy_all_button);
  if ($$props.rtl === void 0 && $$bindings.rtl && rtl !== void 0)
    $$bindings.rtl(rtl);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.avatar_images === void 0 && $$bindings.avatar_images && avatar_images !== void 0)
    $$bindings.avatar_images(avatar_images);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props.render_markdown === void 0 && $$bindings.render_markdown && render_markdown !== void 0)
    $$bindings.render_markdown(render_markdown);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.autoscroll === void 0 && $$bindings.autoscroll && autoscroll !== void 0)
    $$bindings.autoscroll(autoscroll);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.layout === void 0 && $$bindings.layout && layout !== void 0)
    $$bindings.layout(layout);
  if ($$props.placeholder === void 0 && $$bindings.placeholder && placeholder !== void 0)
    $$bindings.placeholder(placeholder);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.msg_format === void 0 && $$bindings.msg_format && msg_format !== void 0)
    $$bindings.msg_format(msg_format);
  if ($$props.examples === void 0 && $$bindings.examples && examples !== void 0)
    $$bindings.examples(examples);
  if ($$props._retryable === void 0 && $$bindings._retryable && _retryable !== void 0)
    $$bindings._retryable(_retryable);
  if ($$props._undoable === void 0 && $$bindings._undoable && _undoable !== void 0)
    $$bindings._undoable(_undoable);
  if ($$props.like_user_message === void 0 && $$bindings.like_user_message && like_user_message !== void 0)
    $$bindings.like_user_message(like_user_message);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  $$result.css.add(css$1);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    {
      update_components();
    }
    {
      if (value || pending_message || _components) {
        scroll_on_value_update();
      }
    }
    {
      {
        if (!dequal(value, old_value)) {
          old_value = value;
          dispatch("change");
        }
      }
    }
    groupedMessages = value && group_messages(value);
    options = value && get_last_bot_options();
    $$rendered = `${value !== null && value.length > 0 ? `${validate_component(IconButtonWrapper, "IconButtonWrapper").$$render($$result, {}, {}, {
      default: () => {
        return `${show_share_button ? `${validate_component(IconButton, "IconButton").$$render($$result, { Icon: Community }, {}, {})}` : ``} ${validate_component(IconButton, "IconButton").$$render($$result, { Icon: Trash, label: "Clear" }, {}, {})} ${show_copy_all_button ? `${validate_component(CopyAll, "CopyAll").$$render($$result, { value }, {}, {})}` : ``}`;
      }
    })}` : ``} <div class="${escape(null_to_empty(layout === "bubble" ? "bubble-wrap" : "panel-wrap"), true) + " svelte-gjtrl6"}" role="log" aria-label="chatbot conversation" aria-live="polite"${add_attribute("this", div, 0)}>${value !== null && value.length > 0 && groupedMessages !== null ? `<div class="message-wrap svelte-gjtrl6">${each(groupedMessages, (messages, i) => {
      let role = messages[0].role === "user" ? "user" : "bot", avatar_img = avatar_images[role === "user" ? 0 : 1], opposite_avatar_img = avatar_images[role === "user" ? 0 : 1], feedback_index = groupedMessages.slice(0, i).filter((m) => m[0].role === "assistant").length, current_feedback = role === "bot" && feedback_value && feedback_value[feedback_index] ? feedback_value[feedback_index] : null;
      return `     ${validate_component(Message, "Message").$$render(
        $$result,
        {
          messages,
          display_consecutive_in_same_bubble,
          opposite_avatar_img,
          avatar_img,
          role,
          layout,
          dispatch,
          i18n,
          _fetch,
          line_breaks,
          theme_mode,
          target,
          root,
          upload,
          selectable,
          sanitize_html,
          render_markdown,
          rtl,
          i,
          value,
          latex_delimiters,
          _components,
          generating,
          msg_format,
          feedback_options,
          current_feedback,
          show_like: role === "user" ? likeable && like_user_message : likeable,
          show_retry: _retryable && is_last_bot_message(messages, value),
          show_undo: _undoable && is_last_bot_message(messages, value),
          show_edit: editable === "all" || editable == "user" && role === "user" && messages.length > 0 && messages[messages.length - 1].type == "text",
          in_edit_mode: edit_index === i,
          show_copy_button,
          handle_action: (selected) => handle_action(i, messages[0], selected),
          scroll: is_browser ? scroll : () => {
          },
          allow_file_downloads,
          edit_message
        },
        {
          edit_message: ($$value) => {
            edit_message = $$value;
            $$settled = false;
          }
        },
        {}
      )}`;
    })} ${pending_message ? `${validate_component(Pending, "Pending").$$render($$result, { layout, avatar_images }, {}, {})}` : `${options ? `<div class="options svelte-gjtrl6">${each(options, (option, index) => {
      return `<button class="option svelte-gjtrl6">${escape(option.label || option.value)} </button>`;
    })}</div>` : ``}`}</div>` : `${validate_component(Examples, "Examples").$$render(
      $$result,
      {
        examples,
        placeholder,
        latex_delimiters,
        root
      },
      {},
      {}
    )}`}</div> ${show_scroll_button ? `<div class="scroll-down-button-container svelte-gjtrl6">${validate_component(IconButton, "IconButton").$$render(
      $$result,
      {
        Icon: ScrollDownArrow,
        label: "Scroll down",
        size: "large"
      },
      {},
      {}
    )}</div>` : ``}`;
  } while (!$$settled);
  return $$rendered;
});
const ChatBot$1 = ChatBot;
const css = {
  code: ".wrapper.svelte-g3p8na{display:flex;position:relative;flex-direction:column;align-items:start;width:100%;height:100%;flex-grow:1}.progress-text{right:auto}",
  map: '{"version":3,"file":"Index.svelte","sources":["Index.svelte"],"sourcesContent":["<script context=\\"module\\" lang=\\"ts\\">export { default as BaseChatBot } from \\"./shared/ChatBot.svelte\\";\\n<\/script>\\n\\n<script lang=\\"ts\\">import ChatBot from \\"./shared/ChatBot.svelte\\";\\nimport { Block, BlockLabel } from \\"@gradio/atoms\\";\\nimport { Chat } from \\"@gradio/icons\\";\\nimport { StatusTracker } from \\"@gradio/statustracker\\";\\nimport { normalise_tuples, normalise_messages } from \\"./shared/utils\\";\\nexport let elem_id = \\"\\";\\nexport let elem_classes = [];\\nexport let visible = true;\\nexport let value = [];\\nexport let scale = null;\\nexport let min_width = void 0;\\nexport let label;\\nexport let show_label = true;\\nexport let root;\\nexport let _selectable = false;\\nexport let likeable = false;\\nexport let feedback_options = [\\"Like\\", \\"Dislike\\"];\\nexport let feedback_value = null;\\nexport let show_share_button = false;\\nexport let rtl = false;\\nexport let show_copy_button = true;\\nexport let show_copy_all_button = false;\\nexport let sanitize_html = true;\\nexport let layout = \\"bubble\\";\\nexport let type = \\"tuples\\";\\nexport let render_markdown = true;\\nexport let line_breaks = true;\\nexport let autoscroll = true;\\nexport let _retryable = false;\\nexport let _undoable = false;\\nexport let group_consecutive_messages = true;\\nexport let latex_delimiters;\\nexport let gradio;\\nlet _value = [];\\n$: _value = type === \\"tuples\\" ? normalise_tuples(value, root) : normalise_messages(value, root);\\nexport let avatar_images = [null, null];\\nexport let like_user_message = false;\\nexport let loading_status = void 0;\\nexport let height;\\nexport let resizeable;\\nexport let min_height;\\nexport let max_height;\\nexport let editable = null;\\nexport let placeholder = null;\\nexport let examples = null;\\nexport let theme_mode;\\nexport let allow_file_downloads = true;\\n<\/script>\\n\\n<Block\\n\\t{elem_id}\\n\\t{elem_classes}\\n\\t{visible}\\n\\tpadding={false}\\n\\t{scale}\\n\\t{min_width}\\n\\t{height}\\n\\t{resizeable}\\n\\t{min_height}\\n\\t{max_height}\\n\\tallow_overflow={true}\\n\\tflex={true}\\n\\toverflow_behavior=\\"auto\\"\\n>\\n\\t{#if loading_status}\\n\\t\\t<StatusTracker\\n\\t\\t\\tautoscroll={gradio.autoscroll}\\n\\t\\t\\ti18n={gradio.i18n}\\n\\t\\t\\t{...loading_status}\\n\\t\\t\\tshow_progress={loading_status.show_progress === \\"hidden\\"\\n\\t\\t\\t\\t? \\"hidden\\"\\n\\t\\t\\t\\t: \\"minimal\\"}\\n\\t\\t\\ton:clear_status={() => gradio.dispatch(\\"clear_status\\", loading_status)}\\n\\t\\t/>\\n\\t{/if}\\n\\t<div class=\\"wrapper\\">\\n\\t\\t{#if show_label}\\n\\t\\t\\t<BlockLabel\\n\\t\\t\\t\\t{show_label}\\n\\t\\t\\t\\tIcon={Chat}\\n\\t\\t\\t\\tfloat={true}\\n\\t\\t\\t\\tlabel={label || \\"Chatbot\\"}\\n\\t\\t\\t/>\\n\\t\\t{/if}\\n\\t\\t<ChatBot\\n\\t\\t\\ti18n={gradio.i18n}\\n\\t\\t\\tselectable={_selectable}\\n\\t\\t\\t{likeable}\\n\\t\\t\\t{feedback_options}\\n\\t\\t\\t{feedback_value}\\n\\t\\t\\t{show_share_button}\\n\\t\\t\\t{show_copy_all_button}\\n\\t\\t\\tvalue={_value}\\n\\t\\t\\t{latex_delimiters}\\n\\t\\t\\tdisplay_consecutive_in_same_bubble={group_consecutive_messages}\\n\\t\\t\\t{render_markdown}\\n\\t\\t\\t{theme_mode}\\n\\t\\t\\t{editable}\\n\\t\\t\\tpending_message={loading_status?.status === \\"pending\\"}\\n\\t\\t\\tgenerating={loading_status?.status === \\"generating\\"}\\n\\t\\t\\t{rtl}\\n\\t\\t\\t{show_copy_button}\\n\\t\\t\\t{like_user_message}\\n\\t\\t\\ton:change={() => gradio.dispatch(\\"change\\", value)}\\n\\t\\t\\ton:select={(e) => gradio.dispatch(\\"select\\", e.detail)}\\n\\t\\t\\ton:like={(e) => gradio.dispatch(\\"like\\", e.detail)}\\n\\t\\t\\ton:share={(e) => gradio.dispatch(\\"share\\", e.detail)}\\n\\t\\t\\ton:error={(e) => gradio.dispatch(\\"error\\", e.detail)}\\n\\t\\t\\ton:example_select={(e) => gradio.dispatch(\\"example_select\\", e.detail)}\\n\\t\\t\\ton:option_select={(e) => gradio.dispatch(\\"option_select\\", e.detail)}\\n\\t\\t\\ton:retry={(e) => gradio.dispatch(\\"retry\\", e.detail)}\\n\\t\\t\\ton:undo={(e) => gradio.dispatch(\\"undo\\", e.detail)}\\n\\t\\t\\ton:clear={() => {\\n\\t\\t\\t\\tvalue = [];\\n\\t\\t\\t\\tgradio.dispatch(\\"clear\\");\\n\\t\\t\\t}}\\n\\t\\t\\ton:copy={(e) => gradio.dispatch(\\"copy\\", e.detail)}\\n\\t\\t\\ton:edit={(e) => {\\n\\t\\t\\t\\tif (value === null || value.length === 0) return;\\n\\t\\t\\t\\tif (type === \\"messages\\") {\\n\\t\\t\\t\\t\\t//@ts-ignore\\n\\t\\t\\t\\t\\tvalue[e.detail.index].content = e.detail.value;\\n\\t\\t\\t\\t} else {\\n\\t\\t\\t\\t\\t//@ts-ignore\\n\\t\\t\\t\\t\\tvalue[e.detail.index[0]][e.detail.index[1]] = e.detail.value;\\n\\t\\t\\t\\t}\\n\\t\\t\\t\\tvalue = value;\\n\\t\\t\\t\\tgradio.dispatch(\\"edit\\", e.detail);\\n\\t\\t\\t}}\\n\\t\\t\\t{avatar_images}\\n\\t\\t\\t{sanitize_html}\\n\\t\\t\\t{line_breaks}\\n\\t\\t\\t{autoscroll}\\n\\t\\t\\t{layout}\\n\\t\\t\\t{placeholder}\\n\\t\\t\\t{examples}\\n\\t\\t\\t{_retryable}\\n\\t\\t\\t{_undoable}\\n\\t\\t\\tupload={(...args) => gradio.client.upload(...args)}\\n\\t\\t\\t_fetch={(...args) => gradio.client.fetch(...args)}\\n\\t\\t\\tload_component={gradio.load_component}\\n\\t\\t\\tmsg_format={type}\\n\\t\\t\\troot={gradio.root}\\n\\t\\t\\t{allow_file_downloads}\\n\\t\\t/>\\n\\t</div>\\n</Block>\\n\\n<style>\\n\\t.wrapper {\\n\\t\\tdisplay: flex;\\n\\t\\tposition: relative;\\n\\t\\tflex-direction: column;\\n\\t\\talign-items: start;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\tflex-grow: 1;\\n\\t}\\n\\n\\t:global(.progress-text) {\\n\\t\\tright: auto;\\n\\t}</style>\\n"],"names":[],"mappings":"AAwJC,sBAAS,CACR,OAAO,CAAE,IAAI,CACb,QAAQ,CAAE,QAAQ,CAClB,cAAc,CAAE,MAAM,CACtB,WAAW,CAAE,KAAK,CAClB,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,SAAS,CAAE,CACZ,CAEQ,cAAgB,CACvB,KAAK,CAAE,IACR"}'
};
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = [] } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { root } = $$props;
  let { _selectable = false } = $$props;
  let { likeable = false } = $$props;
  let { feedback_options = ["Like", "Dislike"] } = $$props;
  let { feedback_value = null } = $$props;
  let { show_share_button = false } = $$props;
  let { rtl = false } = $$props;
  let { show_copy_button = true } = $$props;
  let { show_copy_all_button = false } = $$props;
  let { sanitize_html = true } = $$props;
  let { layout = "bubble" } = $$props;
  let { type = "tuples" } = $$props;
  let { render_markdown = true } = $$props;
  let { line_breaks = true } = $$props;
  let { autoscroll = true } = $$props;
  let { _retryable = false } = $$props;
  let { _undoable = false } = $$props;
  let { group_consecutive_messages = true } = $$props;
  let { latex_delimiters } = $$props;
  let { gradio } = $$props;
  let _value = [];
  let { avatar_images = [null, null] } = $$props;
  let { like_user_message = false } = $$props;
  let { loading_status = void 0 } = $$props;
  let { height } = $$props;
  let { resizeable } = $$props;
  let { min_height } = $$props;
  let { max_height } = $$props;
  let { editable = null } = $$props;
  let { placeholder = null } = $$props;
  let { examples = null } = $$props;
  let { theme_mode } = $$props;
  let { allow_file_downloads = true } = $$props;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props._selectable === void 0 && $$bindings._selectable && _selectable !== void 0)
    $$bindings._selectable(_selectable);
  if ($$props.likeable === void 0 && $$bindings.likeable && likeable !== void 0)
    $$bindings.likeable(likeable);
  if ($$props.feedback_options === void 0 && $$bindings.feedback_options && feedback_options !== void 0)
    $$bindings.feedback_options(feedback_options);
  if ($$props.feedback_value === void 0 && $$bindings.feedback_value && feedback_value !== void 0)
    $$bindings.feedback_value(feedback_value);
  if ($$props.show_share_button === void 0 && $$bindings.show_share_button && show_share_button !== void 0)
    $$bindings.show_share_button(show_share_button);
  if ($$props.rtl === void 0 && $$bindings.rtl && rtl !== void 0)
    $$bindings.rtl(rtl);
  if ($$props.show_copy_button === void 0 && $$bindings.show_copy_button && show_copy_button !== void 0)
    $$bindings.show_copy_button(show_copy_button);
  if ($$props.show_copy_all_button === void 0 && $$bindings.show_copy_all_button && show_copy_all_button !== void 0)
    $$bindings.show_copy_all_button(show_copy_all_button);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props.layout === void 0 && $$bindings.layout && layout !== void 0)
    $$bindings.layout(layout);
  if ($$props.type === void 0 && $$bindings.type && type !== void 0)
    $$bindings.type(type);
  if ($$props.render_markdown === void 0 && $$bindings.render_markdown && render_markdown !== void 0)
    $$bindings.render_markdown(render_markdown);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.autoscroll === void 0 && $$bindings.autoscroll && autoscroll !== void 0)
    $$bindings.autoscroll(autoscroll);
  if ($$props._retryable === void 0 && $$bindings._retryable && _retryable !== void 0)
    $$bindings._retryable(_retryable);
  if ($$props._undoable === void 0 && $$bindings._undoable && _undoable !== void 0)
    $$bindings._undoable(_undoable);
  if ($$props.group_consecutive_messages === void 0 && $$bindings.group_consecutive_messages && group_consecutive_messages !== void 0)
    $$bindings.group_consecutive_messages(group_consecutive_messages);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.avatar_images === void 0 && $$bindings.avatar_images && avatar_images !== void 0)
    $$bindings.avatar_images(avatar_images);
  if ($$props.like_user_message === void 0 && $$bindings.like_user_message && like_user_message !== void 0)
    $$bindings.like_user_message(like_user_message);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.resizeable === void 0 && $$bindings.resizeable && resizeable !== void 0)
    $$bindings.resizeable(resizeable);
  if ($$props.min_height === void 0 && $$bindings.min_height && min_height !== void 0)
    $$bindings.min_height(min_height);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.placeholder === void 0 && $$bindings.placeholder && placeholder !== void 0)
    $$bindings.placeholder(placeholder);
  if ($$props.examples === void 0 && $$bindings.examples && examples !== void 0)
    $$bindings.examples(examples);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props.allow_file_downloads === void 0 && $$bindings.allow_file_downloads && allow_file_downloads !== void 0)
    $$bindings.allow_file_downloads(allow_file_downloads);
  $$result.css.add(css);
  _value = type === "tuples" ? normalise_tuples(value, root) : normalise_messages(value, root);
  return `${validate_component(Block, "Block").$$render(
    $$result,
    {
      elem_id,
      elem_classes,
      visible,
      padding: false,
      scale,
      min_width,
      height,
      resizeable,
      min_height,
      max_height,
      allow_overflow: true,
      flex: true,
      overflow_behavior: "auto"
    },
    {},
    {
      default: () => {
        return `${loading_status ? `${validate_component(Static, "StatusTracker").$$render(
          $$result,
          Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status, {
            show_progress: loading_status.show_progress === "hidden" ? "hidden" : "minimal"
          }),
          {},
          {}
        )}` : ``} <div class="wrapper svelte-g3p8na">${show_label ? `${validate_component(BlockLabel, "BlockLabel").$$render(
          $$result,
          {
            show_label,
            Icon: Chat,
            float: true,
            label: label || "Chatbot"
          },
          {},
          {}
        )}` : ``} ${validate_component(ChatBot$1, "ChatBot").$$render(
          $$result,
          {
            i18n: gradio.i18n,
            selectable: _selectable,
            likeable,
            feedback_options,
            feedback_value,
            show_share_button,
            show_copy_all_button,
            value: _value,
            latex_delimiters,
            display_consecutive_in_same_bubble: group_consecutive_messages,
            render_markdown,
            theme_mode,
            editable,
            pending_message: loading_status?.status === "pending",
            generating: loading_status?.status === "generating",
            rtl,
            show_copy_button,
            like_user_message,
            avatar_images,
            sanitize_html,
            line_breaks,
            autoscroll,
            layout,
            placeholder,
            examples,
            _retryable,
            _undoable,
            upload: (...args) => gradio.client.upload(...args),
            _fetch: (...args) => gradio.client.fetch(...args),
            load_component: gradio.load_component,
            msg_format: type,
            root: gradio.root,
            allow_file_downloads
          },
          {},
          {}
        )}</div>`;
      }
    }
  )}`;
});

export { ChatBot$1 as BaseChatBot, Index as default };
//# sourceMappingURL=Index57-CtKpUUZX.js.map
