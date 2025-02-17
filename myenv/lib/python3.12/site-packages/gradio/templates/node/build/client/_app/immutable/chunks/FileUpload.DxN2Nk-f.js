import { SvelteComponent, init, safe_not_equal, ensure_array_like, element, claim_element, children, detach, attr, set_style, insert_hydration, append_hydration, group_outros, update_keyed_each, outro_and_destroy_block, check_outros, transition_in, transition_out, createEventDispatcher, space, text, claim_space, claim_text, toggle_class, listen, prevent_default, set_data, run_all, bubble, get_svelte_dataset, noop, create_component, claim_component, mount_component, destroy_component, HtmlTagHydration, claim_html_tag, empty, tick, binding_callbacks, bind, add_flush_callback, create_slot, update_slot_base, get_all_dirty_from_scope, get_slot_changes } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { I as IconButton, C as Clear } from "./2.8WKXZUMv.js";
import { B as BlockLabel } from "./BlockLabel.BcwqJKEo.js";
import { E as Empty } from "./Empty.DwcIP_BA.js";
import { F as File$1 } from "./File.DqOJDDoa.js";
import { U as Upload } from "./Upload.C9nHsuEq.js";
import { U as Upload$1 } from "./Upload.v2Thvwuk.js";
import { I as IconButtonWrapper } from "./IconButtonWrapper.CePV_r65.js";
/* empty css                                                    */
import { D as DownloadLink } from "./DownloadLink.CzZp0moC.js";
const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + "&nbsp;" + unit;
};
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[25] = list[i];
  child_ctx[27] = i;
  return child_ctx;
}
function create_if_block_2(ctx) {
  let span;
  let textContent = "⋮⋮";
  return {
    c() {
      span = element("span");
      span.textContent = textContent;
      this.h();
    },
    l(nodes) {
      span = claim_element(nodes, "SPAN", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(span) !== "svelte-1u4up0a")
        span.textContent = textContent;
      this.h();
    },
    h() {
      attr(span, "class", "drag-handle svelte-1rvzbk6");
    },
    m(target, anchor) {
      insert_hydration(target, span, anchor);
    },
    d(detaching) {
      if (detaching) {
        detach(span);
      }
    }
  };
}
function create_else_block$2(ctx) {
  let t_value = (
    /*i18n*/
    ctx[2]("file.uploading") + ""
  );
  let t;
  return {
    c() {
      t = text(t_value);
    },
    l(nodes) {
      t = claim_text(nodes, t_value);
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty & /*i18n*/
      4 && t_value !== (t_value = /*i18n*/
      ctx2[2]("file.uploading") + ""))
        set_data(t, t_value);
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_1$1(ctx) {
  let downloadlink;
  let current;
  function click_handler() {
    return (
      /*click_handler*/
      ctx[17](
        /*file*/
        ctx[25]
      )
    );
  }
  downloadlink = new DownloadLink({
    props: {
      href: (
        /*file*/
        ctx[25].url
      ),
      download: (
        /*is_browser*/
        ctx[14] && window.__is_colab__ ? null : (
          /*file*/
          ctx[25].orig_name
        )
      ),
      $$slots: { default: [create_default_slot$2] },
      $$scope: { ctx }
    }
  });
  downloadlink.$on("click", click_handler);
  return {
    c() {
      create_component(downloadlink.$$.fragment);
    },
    l(nodes) {
      claim_component(downloadlink.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(downloadlink, target, anchor);
      current = true;
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      const downloadlink_changes = {};
      if (dirty & /*normalized_files*/
      64)
        downloadlink_changes.href = /*file*/
        ctx[25].url;
      if (dirty & /*normalized_files*/
      64)
        downloadlink_changes.download = /*is_browser*/
        ctx[14] && window.__is_colab__ ? null : (
          /*file*/
          ctx[25].orig_name
        );
      if (dirty & /*$$scope, normalized_files*/
      268435520) {
        downloadlink_changes.$$scope = { dirty, ctx };
      }
      downloadlink.$set(downloadlink_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(downloadlink.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(downloadlink.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(downloadlink, detaching);
    }
  };
}
function create_default_slot$2(ctx) {
  let html_tag;
  let raw_value = (
    /*file*/
    (ctx[25].size != null ? prettyBytes(
      /*file*/
      ctx[25].size
    ) : "(size unknown)") + ""
  );
  let t;
  return {
    c() {
      html_tag = new HtmlTagHydration(false);
      t = text(" ⇣");
      this.h();
    },
    l(nodes) {
      html_tag = claim_html_tag(nodes, false);
      t = claim_text(nodes, " ⇣");
      this.h();
    },
    h() {
      html_tag.a = t;
    },
    m(target, anchor) {
      html_tag.m(raw_value, target, anchor);
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty & /*normalized_files*/
      64 && raw_value !== (raw_value = /*file*/
      (ctx2[25].size != null ? prettyBytes(
        /*file*/
        ctx2[25].size
      ) : "(size unknown)") + ""))
        html_tag.p(raw_value);
    },
    d(detaching) {
      if (detaching) {
        html_tag.d();
        detach(t);
      }
    }
  };
}
function create_if_block$2(ctx) {
  let td;
  let button;
  let textContent = "×";
  let mounted;
  let dispose;
  function click_handler_1() {
    return (
      /*click_handler_1*/
      ctx[18](
        /*i*/
        ctx[27]
      )
    );
  }
  function keydown_handler(...args) {
    return (
      /*keydown_handler*/
      ctx[19](
        /*i*/
        ctx[27],
        ...args
      )
    );
  }
  return {
    c() {
      td = element("td");
      button = element("button");
      button.textContent = textContent;
      this.h();
    },
    l(nodes) {
      td = claim_element(nodes, "TD", { class: true });
      var td_nodes = children(td);
      button = claim_element(td_nodes, "BUTTON", {
        class: true,
        "aria-label": true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button) !== "svelte-nhtord")
        button.textContent = textContent;
      td_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "label-clear-button svelte-1rvzbk6");
      attr(button, "aria-label", "Remove this file");
      attr(td, "class", "svelte-1rvzbk6");
    },
    m(target, anchor) {
      insert_hydration(target, td, anchor);
      append_hydration(td, button);
      if (!mounted) {
        dispose = [
          listen(button, "click", click_handler_1),
          listen(button, "keydown", keydown_handler)
        ];
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
    },
    d(detaching) {
      if (detaching) {
        detach(td);
      }
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_each_block(key_1, ctx) {
  let tr;
  let td0;
  let t0;
  let span0;
  let t1_value = (
    /*file*/
    ctx[25].filename_stem + ""
  );
  let t1;
  let t2;
  let span1;
  let t3_value = (
    /*file*/
    ctx[25].filename_ext + ""
  );
  let t3;
  let td0_aria_label_value;
  let t4;
  let td1;
  let current_block_type_index;
  let if_block1;
  let t5;
  let t6;
  let tr_data_drop_target_value;
  let tr_draggable_value;
  let current;
  let mounted;
  let dispose;
  let if_block0 = (
    /*allow_reordering*/
    ctx[3] && /*normalized_files*/
    ctx[6].length > 1 && create_if_block_2()
  );
  const if_block_creators = [create_if_block_1$1, create_else_block$2];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*file*/
      ctx2[25].url
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block1 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  let if_block2 = (
    /*normalized_files*/
    ctx[6].length > 1 && create_if_block$2(ctx)
  );
  function click_handler_2(...args) {
    return (
      /*click_handler_2*/
      ctx[20](
        /*i*/
        ctx[27],
        ...args
      )
    );
  }
  function dragstart_handler(...args) {
    return (
      /*dragstart_handler*/
      ctx[21](
        /*i*/
        ctx[27],
        ...args
      )
    );
  }
  function dragover_handler(...args) {
    return (
      /*dragover_handler*/
      ctx[22](
        /*i*/
        ctx[27],
        ...args
      )
    );
  }
  function drop_handler(...args) {
    return (
      /*drop_handler*/
      ctx[23](
        /*i*/
        ctx[27],
        ...args
      )
    );
  }
  return {
    key: key_1,
    first: null,
    c() {
      tr = element("tr");
      td0 = element("td");
      if (if_block0)
        if_block0.c();
      t0 = space();
      span0 = element("span");
      t1 = text(t1_value);
      t2 = space();
      span1 = element("span");
      t3 = text(t3_value);
      t4 = space();
      td1 = element("td");
      if_block1.c();
      t5 = space();
      if (if_block2)
        if_block2.c();
      t6 = space();
      this.h();
    },
    l(nodes) {
      tr = claim_element(nodes, "TR", {
        class: true,
        "data-drop-target": true,
        draggable: true
      });
      var tr_nodes = children(tr);
      td0 = claim_element(tr_nodes, "TD", { class: true, "aria-label": true });
      var td0_nodes = children(td0);
      if (if_block0)
        if_block0.l(td0_nodes);
      t0 = claim_space(td0_nodes);
      span0 = claim_element(td0_nodes, "SPAN", { class: true });
      var span0_nodes = children(span0);
      t1 = claim_text(span0_nodes, t1_value);
      span0_nodes.forEach(detach);
      t2 = claim_space(td0_nodes);
      span1 = claim_element(td0_nodes, "SPAN", { class: true });
      var span1_nodes = children(span1);
      t3 = claim_text(span1_nodes, t3_value);
      span1_nodes.forEach(detach);
      td0_nodes.forEach(detach);
      t4 = claim_space(tr_nodes);
      td1 = claim_element(tr_nodes, "TD", { class: true });
      var td1_nodes = children(td1);
      if_block1.l(td1_nodes);
      td1_nodes.forEach(detach);
      t5 = claim_space(tr_nodes);
      if (if_block2)
        if_block2.l(tr_nodes);
      t6 = claim_space(tr_nodes);
      tr_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(span0, "class", "stem svelte-1rvzbk6");
      attr(span1, "class", "ext svelte-1rvzbk6");
      attr(td0, "class", "filename svelte-1rvzbk6");
      attr(td0, "aria-label", td0_aria_label_value = /*file*/
      ctx[25].orig_name);
      attr(td1, "class", "download svelte-1rvzbk6");
      attr(tr, "class", "file svelte-1rvzbk6");
      attr(tr, "data-drop-target", tr_data_drop_target_value = /*drop_target_index*/
      ctx[5] === /*normalized_files*/
      ctx[6].length && /*i*/
      ctx[27] === /*normalized_files*/
      ctx[6].length - 1 ? "after" : (
        /*drop_target_index*/
        ctx[5] === /*i*/
        ctx[27] + 1 ? "after" : "before"
      ));
      attr(tr, "draggable", tr_draggable_value = /*allow_reordering*/
      ctx[3] && /*normalized_files*/
      ctx[6].length > 1);
      toggle_class(
        tr,
        "selectable",
        /*selectable*/
        ctx[0]
      );
      toggle_class(
        tr,
        "dragging",
        /*dragging_index*/
        ctx[4] === /*i*/
        ctx[27]
      );
      toggle_class(
        tr,
        "drop-target",
        /*drop_target_index*/
        ctx[5] === /*i*/
        ctx[27] || /*i*/
        ctx[27] === /*normalized_files*/
        ctx[6].length - 1 && /*drop_target_index*/
        ctx[5] === /*normalized_files*/
        ctx[6].length
      );
      this.first = tr;
    },
    m(target, anchor) {
      insert_hydration(target, tr, anchor);
      append_hydration(tr, td0);
      if (if_block0)
        if_block0.m(td0, null);
      append_hydration(td0, t0);
      append_hydration(td0, span0);
      append_hydration(span0, t1);
      append_hydration(td0, t2);
      append_hydration(td0, span1);
      append_hydration(span1, t3);
      append_hydration(tr, t4);
      append_hydration(tr, td1);
      if_blocks[current_block_type_index].m(td1, null);
      append_hydration(tr, t5);
      if (if_block2)
        if_block2.m(tr, null);
      append_hydration(tr, t6);
      current = true;
      if (!mounted) {
        dispose = [
          listen(tr, "click", click_handler_2),
          listen(tr, "dragstart", dragstart_handler),
          listen(tr, "dragenter", prevent_default(
            /*dragenter_handler*/
            ctx[16]
          )),
          listen(tr, "dragover", dragover_handler),
          listen(tr, "drop", drop_handler),
          listen(
            tr,
            "dragend",
            /*handle_drag_end*/
            ctx[9]
          )
        ];
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (
        /*allow_reordering*/
        ctx[3] && /*normalized_files*/
        ctx[6].length > 1
      ) {
        if (if_block0)
          ;
        else {
          if_block0 = create_if_block_2();
          if_block0.c();
          if_block0.m(td0, t0);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      if ((!current || dirty & /*normalized_files*/
      64) && t1_value !== (t1_value = /*file*/
      ctx[25].filename_stem + ""))
        set_data(t1, t1_value);
      if ((!current || dirty & /*normalized_files*/
      64) && t3_value !== (t3_value = /*file*/
      ctx[25].filename_ext + ""))
        set_data(t3, t3_value);
      if (!current || dirty & /*normalized_files*/
      64 && td0_aria_label_value !== (td0_aria_label_value = /*file*/
      ctx[25].orig_name)) {
        attr(td0, "aria-label", td0_aria_label_value);
      }
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block1 = if_blocks[current_block_type_index];
        if (!if_block1) {
          if_block1 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
          if_block1.c();
        } else {
          if_block1.p(ctx, dirty);
        }
        transition_in(if_block1, 1);
        if_block1.m(td1, null);
      }
      if (
        /*normalized_files*/
        ctx[6].length > 1
      ) {
        if (if_block2) {
          if_block2.p(ctx, dirty);
        } else {
          if_block2 = create_if_block$2(ctx);
          if_block2.c();
          if_block2.m(tr, t6);
        }
      } else if (if_block2) {
        if_block2.d(1);
        if_block2 = null;
      }
      if (!current || dirty & /*drop_target_index, normalized_files*/
      96 && tr_data_drop_target_value !== (tr_data_drop_target_value = /*drop_target_index*/
      ctx[5] === /*normalized_files*/
      ctx[6].length && /*i*/
      ctx[27] === /*normalized_files*/
      ctx[6].length - 1 ? "after" : (
        /*drop_target_index*/
        ctx[5] === /*i*/
        ctx[27] + 1 ? "after" : "before"
      ))) {
        attr(tr, "data-drop-target", tr_data_drop_target_value);
      }
      if (!current || dirty & /*allow_reordering, normalized_files*/
      72 && tr_draggable_value !== (tr_draggable_value = /*allow_reordering*/
      ctx[3] && /*normalized_files*/
      ctx[6].length > 1)) {
        attr(tr, "draggable", tr_draggable_value);
      }
      if (!current || dirty & /*selectable*/
      1) {
        toggle_class(
          tr,
          "selectable",
          /*selectable*/
          ctx[0]
        );
      }
      if (!current || dirty & /*dragging_index, normalized_files*/
      80) {
        toggle_class(
          tr,
          "dragging",
          /*dragging_index*/
          ctx[4] === /*i*/
          ctx[27]
        );
      }
      if (!current || dirty & /*drop_target_index, normalized_files*/
      96) {
        toggle_class(
          tr,
          "drop-target",
          /*drop_target_index*/
          ctx[5] === /*i*/
          ctx[27] || /*i*/
          ctx[27] === /*normalized_files*/
          ctx[6].length - 1 && /*drop_target_index*/
          ctx[5] === /*normalized_files*/
          ctx[6].length
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(tr);
      }
      if (if_block0)
        if_block0.d();
      if_blocks[current_block_type_index].d();
      if (if_block2)
        if_block2.d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_fragment$2(ctx) {
  let div;
  let table;
  let tbody;
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let current;
  let each_value = ensure_array_like(
    /*normalized_files*/
    ctx[6]
  );
  const get_key = (ctx2) => (
    /*file*/
    ctx2[25].url
  );
  for (let i = 0; i < each_value.length; i += 1) {
    let child_ctx = get_each_context(ctx, each_value, i);
    let key = get_key(child_ctx);
    each_1_lookup.set(key, each_blocks[i] = create_each_block(key, child_ctx));
  }
  return {
    c() {
      div = element("div");
      table = element("table");
      tbody = element("tbody");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      table = claim_element(div_nodes, "TABLE", { class: true });
      var table_nodes = children(table);
      tbody = claim_element(table_nodes, "TBODY", { class: true });
      var tbody_nodes = children(tbody);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(tbody_nodes);
      }
      tbody_nodes.forEach(detach);
      table_nodes.forEach(detach);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(tbody, "class", "svelte-1rvzbk6");
      attr(table, "class", "file-preview svelte-1rvzbk6");
      attr(div, "class", "file-preview-holder svelte-1rvzbk6");
      set_style(
        div,
        "max-height",
        /*height*/
        ctx[1] ? typeof /*height*/
        ctx[1] === "number" ? (
          /*height*/
          ctx[1] + "px"
        ) : (
          /*height*/
          ctx[1]
        ) : "auto"
      );
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      append_hydration(div, table);
      append_hydration(table, tbody);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(tbody, null);
        }
      }
      current = true;
    },
    p(ctx2, [dirty]) {
      if (dirty & /*drop_target_index, normalized_files, allow_reordering, selectable, dragging_index, handle_row_click, handle_drag_start, handle_drag_over, handle_drop, handle_drag_end, remove_file, is_browser, window, handle_download, i18n*/
      32765) {
        each_value = ensure_array_like(
          /*normalized_files*/
          ctx2[6]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value, each_1_lookup, tbody, outro_and_destroy_block, create_each_block, null, get_each_context);
        check_outros();
      }
      if (dirty & /*height*/
      2) {
        set_style(
          div,
          "max-height",
          /*height*/
          ctx2[1] ? typeof /*height*/
          ctx2[1] === "number" ? (
            /*height*/
            ctx2[1] + "px"
          ) : (
            /*height*/
            ctx2[1]
          ) : "auto"
        );
      }
    },
    i(local) {
      if (current)
        return;
      for (let i = 0; i < each_value.length; i += 1) {
        transition_in(each_blocks[i]);
      }
      current = true;
    },
    o(local) {
      for (let i = 0; i < each_blocks.length; i += 1) {
        transition_out(each_blocks[i]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d();
      }
    }
  };
}
function split_filename(filename) {
  const last_dot = filename.lastIndexOf(".");
  if (last_dot === -1) {
    return [filename, ""];
  }
  return [filename.slice(0, last_dot), filename.slice(last_dot)];
}
function instance$2($$self, $$props, $$invalidate) {
  let normalized_files;
  const dispatch = createEventDispatcher();
  let { value } = $$props;
  let { selectable = false } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  let { allow_reordering = false } = $$props;
  let dragging_index = null;
  let drop_target_index = null;
  function handle_drag_start(event, index) {
    $$invalidate(4, dragging_index = index);
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", index.toString());
    }
  }
  function handle_drag_over(event, index) {
    event.preventDefault();
    if (index === normalized_files.length - 1) {
      const rect = event.currentTarget.getBoundingClientRect();
      const midY = rect.top + rect.height / 2;
      $$invalidate(5, drop_target_index = event.clientY > midY ? normalized_files.length : index);
    } else {
      $$invalidate(5, drop_target_index = index);
    }
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "move";
    }
  }
  function handle_drag_end(event) {
    var _a;
    if (!((_a = event.dataTransfer) == null ? void 0 : _a.dropEffect) || event.dataTransfer.dropEffect === "none") {
      $$invalidate(4, dragging_index = null);
      $$invalidate(5, drop_target_index = null);
    }
  }
  function handle_drop(event, index) {
    event.preventDefault();
    if (dragging_index === null || dragging_index === index)
      return;
    const files = Array.isArray(value) ? [...value] : [value];
    const [removed] = files.splice(dragging_index, 1);
    files.splice(
      drop_target_index === normalized_files.length ? normalized_files.length : index,
      0,
      removed
    );
    const new_value = Array.isArray(value) ? files : files[0];
    dispatch("change", new_value);
    $$invalidate(4, dragging_index = null);
    $$invalidate(5, drop_target_index = null);
  }
  function handle_row_click(event, index) {
    const tr = event.currentTarget;
    const should_select = event.target === tr || // Only select if the click is on the row itself
    tr && tr.firstElementChild && event.composedPath().includes(tr.firstElementChild);
    if (should_select) {
      dispatch("select", {
        value: normalized_files[index].orig_name,
        index
      });
    }
  }
  function remove_file(index) {
    const removed = normalized_files.splice(index, 1);
    $$invalidate(6, normalized_files = [...normalized_files]);
    $$invalidate(15, value = normalized_files);
    dispatch("delete", removed[0]);
    dispatch("change", normalized_files);
  }
  function handle_download(file) {
    dispatch("download", file);
  }
  const is_browser = typeof window !== "undefined";
  function dragenter_handler(event) {
    bubble.call(this, $$self, event);
  }
  const click_handler = (file) => handle_download(file);
  const click_handler_1 = (i) => {
    remove_file(i);
  };
  const keydown_handler = (i, event) => {
    if (event.key === "Enter") {
      remove_file(i);
    }
  };
  const click_handler_2 = (i, event) => {
    handle_row_click(event, i);
  };
  const dragstart_handler = (i, event) => handle_drag_start(event, i);
  const dragover_handler = (i, event) => handle_drag_over(event, i);
  const drop_handler = (i, event) => handle_drop(event, i);
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(15, value = $$props2.value);
    if ("selectable" in $$props2)
      $$invalidate(0, selectable = $$props2.selectable);
    if ("height" in $$props2)
      $$invalidate(1, height = $$props2.height);
    if ("i18n" in $$props2)
      $$invalidate(2, i18n = $$props2.i18n);
    if ("allow_reordering" in $$props2)
      $$invalidate(3, allow_reordering = $$props2.allow_reordering);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value*/
    32768) {
      $$invalidate(6, normalized_files = (Array.isArray(value) ? value : [value]).map((file) => {
        const [filename_stem, filename_ext] = split_filename(file.orig_name ?? "");
        return { ...file, filename_stem, filename_ext };
      }));
    }
  };
  return [
    selectable,
    height,
    i18n,
    allow_reordering,
    dragging_index,
    drop_target_index,
    normalized_files,
    handle_drag_start,
    handle_drag_over,
    handle_drag_end,
    handle_drop,
    handle_row_click,
    remove_file,
    handle_download,
    is_browser,
    value,
    dragenter_handler,
    click_handler,
    click_handler_1,
    keydown_handler,
    click_handler_2,
    dragstart_handler,
    dragover_handler,
    drop_handler
  ];
}
class FilePreview extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$2, create_fragment$2, safe_not_equal, {
      value: 15,
      selectable: 0,
      height: 1,
      i18n: 2,
      allow_reordering: 3
    });
  }
}
const FilePreview$1 = FilePreview;
function create_else_block$1(ctx) {
  let empty_1;
  let current;
  empty_1 = new Empty({
    props: {
      unpadded_box: true,
      size: "large",
      $$slots: { default: [create_default_slot$1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(empty_1.$$.fragment);
    },
    l(nodes) {
      claim_component(empty_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(empty_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const empty_1_changes = {};
      if (dirty & /*$$scope*/
      256) {
        empty_1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      empty_1.$set(empty_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(empty_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(empty_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(empty_1, detaching);
    }
  };
}
function create_if_block$1(ctx) {
  let filepreview;
  let current;
  filepreview = new FilePreview$1({
    props: {
      i18n: (
        /*i18n*/
        ctx[5]
      ),
      selectable: (
        /*selectable*/
        ctx[3]
      ),
      value: (
        /*value*/
        ctx[0]
      ),
      height: (
        /*height*/
        ctx[4]
      )
    }
  });
  filepreview.$on(
    "select",
    /*select_handler*/
    ctx[6]
  );
  filepreview.$on(
    "download",
    /*download_handler*/
    ctx[7]
  );
  return {
    c() {
      create_component(filepreview.$$.fragment);
    },
    l(nodes) {
      claim_component(filepreview.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(filepreview, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const filepreview_changes = {};
      if (dirty & /*i18n*/
      32)
        filepreview_changes.i18n = /*i18n*/
        ctx2[5];
      if (dirty & /*selectable*/
      8)
        filepreview_changes.selectable = /*selectable*/
        ctx2[3];
      if (dirty & /*value*/
      1)
        filepreview_changes.value = /*value*/
        ctx2[0];
      if (dirty & /*height*/
      16)
        filepreview_changes.height = /*height*/
        ctx2[4];
      filepreview.$set(filepreview_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(filepreview.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(filepreview.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(filepreview, detaching);
    }
  };
}
function create_default_slot$1(ctx) {
  let file;
  let current;
  file = new File$1({});
  return {
    c() {
      create_component(file.$$.fragment);
    },
    l(nodes) {
      claim_component(file.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(file, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(file.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(file.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(file, detaching);
    }
  };
}
function create_fragment$1(ctx) {
  let blocklabel;
  let t;
  let show_if;
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  blocklabel = new BlockLabel({
    props: {
      show_label: (
        /*show_label*/
        ctx[2]
      ),
      float: (
        /*value*/
        ctx[0] === null
      ),
      Icon: File$1,
      label: (
        /*label*/
        ctx[1] || "File"
      )
    }
  });
  const if_block_creators = [create_if_block$1, create_else_block$1];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (dirty & /*value*/
    1)
      show_if = null;
    if (show_if == null)
      show_if = !!/*value*/
      (ctx2[0] && (Array.isArray(
        /*value*/
        ctx2[0]
      ) ? (
        /*value*/
        ctx2[0].length > 0
      ) : true));
    if (show_if)
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx, -1);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(blocklabel.$$.fragment);
      t = space();
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
      t = claim_space(nodes);
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      insert_hydration(target, t, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const blocklabel_changes = {};
      if (dirty & /*show_label*/
      4)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[2];
      if (dirty & /*value*/
      1)
        blocklabel_changes.float = /*value*/
        ctx2[0] === null;
      if (dirty & /*label*/
      2)
        blocklabel_changes.label = /*label*/
        ctx2[1] || "File";
      blocklabel.$set(blocklabel_changes);
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2, dirty);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        } else {
          if_block.p(ctx2, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(if_block_anchor.parentNode, if_block_anchor);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(blocklabel.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block_anchor);
      }
      destroy_component(blocklabel, detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let { value = null } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { selectable = false } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  function select_handler(event) {
    bubble.call(this, $$self, event);
  }
  function download_handler(event) {
    bubble.call(this, $$self, event);
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("label" in $$props2)
      $$invalidate(1, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(2, show_label = $$props2.show_label);
    if ("selectable" in $$props2)
      $$invalidate(3, selectable = $$props2.selectable);
    if ("height" in $$props2)
      $$invalidate(4, height = $$props2.height);
    if ("i18n" in $$props2)
      $$invalidate(5, i18n = $$props2.i18n);
  };
  return [
    value,
    label,
    show_label,
    selectable,
    height,
    i18n,
    select_handler,
    download_handler
  ];
}
class File_1 extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$1, create_fragment$1, safe_not_equal, {
      value: 0,
      label: 1,
      show_label: 2,
      selectable: 3,
      height: 4,
      i18n: 5
    });
  }
}
const File = File_1;
function create_else_block(ctx) {
  let upload_1;
  let updating_dragging;
  let updating_uploading;
  let current;
  function upload_1_dragging_binding_1(value) {
    ctx[26](value);
  }
  function upload_1_uploading_binding_1(value) {
    ctx[27](value);
  }
  let upload_1_props = {
    filetype: (
      /*file_types*/
      ctx[5]
    ),
    file_count: (
      /*file_count*/
      ctx[4]
    ),
    max_file_size: (
      /*max_file_size*/
      ctx[10]
    ),
    root: (
      /*root*/
      ctx[7]
    ),
    stream_handler: (
      /*stream_handler*/
      ctx[12]
    ),
    upload: (
      /*upload*/
      ctx[11]
    ),
    height: (
      /*height*/
      ctx[8]
    ),
    $$slots: { default: [create_default_slot_2] },
    $$scope: { ctx }
  };
  if (
    /*dragging*/
    ctx[14] !== void 0
  ) {
    upload_1_props.dragging = /*dragging*/
    ctx[14];
  }
  if (
    /*uploading*/
    ctx[1] !== void 0
  ) {
    upload_1_props.uploading = /*uploading*/
    ctx[1];
  }
  upload_1 = new Upload({ props: upload_1_props });
  binding_callbacks.push(() => bind(upload_1, "dragging", upload_1_dragging_binding_1));
  binding_callbacks.push(() => bind(upload_1, "uploading", upload_1_uploading_binding_1));
  upload_1.$on(
    "load",
    /*handle_upload*/
    ctx[15]
  );
  upload_1.$on(
    "error",
    /*error_handler_1*/
    ctx[28]
  );
  return {
    c() {
      create_component(upload_1.$$.fragment);
    },
    l(nodes) {
      claim_component(upload_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(upload_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const upload_1_changes = {};
      if (dirty & /*file_types*/
      32)
        upload_1_changes.filetype = /*file_types*/
        ctx2[5];
      if (dirty & /*file_count*/
      16)
        upload_1_changes.file_count = /*file_count*/
        ctx2[4];
      if (dirty & /*max_file_size*/
      1024)
        upload_1_changes.max_file_size = /*max_file_size*/
        ctx2[10];
      if (dirty & /*root*/
      128)
        upload_1_changes.root = /*root*/
        ctx2[7];
      if (dirty & /*stream_handler*/
      4096)
        upload_1_changes.stream_handler = /*stream_handler*/
        ctx2[12];
      if (dirty & /*upload*/
      2048)
        upload_1_changes.upload = /*upload*/
        ctx2[11];
      if (dirty & /*height*/
      256)
        upload_1_changes.height = /*height*/
        ctx2[8];
      if (dirty & /*$$scope*/
      536870912) {
        upload_1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_dragging && dirty & /*dragging*/
      16384) {
        updating_dragging = true;
        upload_1_changes.dragging = /*dragging*/
        ctx2[14];
        add_flush_callback(() => updating_dragging = false);
      }
      if (!updating_uploading && dirty & /*uploading*/
      2) {
        updating_uploading = true;
        upload_1_changes.uploading = /*uploading*/
        ctx2[1];
        add_flush_callback(() => updating_uploading = false);
      }
      upload_1.$set(upload_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(upload_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(upload_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(upload_1, detaching);
    }
  };
}
function create_if_block(ctx) {
  let iconbuttonwrapper;
  let t;
  let filepreview;
  let current;
  iconbuttonwrapper = new IconButtonWrapper({
    props: {
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  filepreview = new FilePreview$1({
    props: {
      i18n: (
        /*i18n*/
        ctx[9]
      ),
      selectable: (
        /*selectable*/
        ctx[6]
      ),
      value: (
        /*value*/
        ctx[0]
      ),
      height: (
        /*height*/
        ctx[8]
      ),
      allow_reordering: (
        /*allow_reordering*/
        ctx[13]
      )
    }
  });
  filepreview.$on(
    "select",
    /*select_handler*/
    ctx[23]
  );
  filepreview.$on(
    "change",
    /*change_handler*/
    ctx[24]
  );
  filepreview.$on(
    "delete",
    /*delete_handler*/
    ctx[25]
  );
  return {
    c() {
      create_component(iconbuttonwrapper.$$.fragment);
      t = space();
      create_component(filepreview.$$.fragment);
    },
    l(nodes) {
      claim_component(iconbuttonwrapper.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(filepreview.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(iconbuttonwrapper, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(filepreview, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const iconbuttonwrapper_changes = {};
      if (dirty & /*$$scope, i18n, file_types, file_count, max_file_size, root, stream_handler, upload, dragging, uploading, value*/
      536895155) {
        iconbuttonwrapper_changes.$$scope = { dirty, ctx: ctx2 };
      }
      iconbuttonwrapper.$set(iconbuttonwrapper_changes);
      const filepreview_changes = {};
      if (dirty & /*i18n*/
      512)
        filepreview_changes.i18n = /*i18n*/
        ctx2[9];
      if (dirty & /*selectable*/
      64)
        filepreview_changes.selectable = /*selectable*/
        ctx2[6];
      if (dirty & /*value*/
      1)
        filepreview_changes.value = /*value*/
        ctx2[0];
      if (dirty & /*height*/
      256)
        filepreview_changes.height = /*height*/
        ctx2[8];
      if (dirty & /*allow_reordering*/
      8192)
        filepreview_changes.allow_reordering = /*allow_reordering*/
        ctx2[13];
      filepreview.$set(filepreview_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(iconbuttonwrapper.$$.fragment, local);
      transition_in(filepreview.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(iconbuttonwrapper.$$.fragment, local);
      transition_out(filepreview.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(iconbuttonwrapper, detaching);
      destroy_component(filepreview, detaching);
    }
  };
}
function create_default_slot_2(ctx) {
  let current;
  const default_slot_template = (
    /*#slots*/
    ctx[18].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[29],
    null
  );
  return {
    c() {
      if (default_slot)
        default_slot.c();
    },
    l(nodes) {
      if (default_slot)
        default_slot.l(nodes);
    },
    m(target, anchor) {
      if (default_slot) {
        default_slot.m(target, anchor);
      }
      current = true;
    },
    p(ctx2, dirty) {
      if (default_slot) {
        if (default_slot.p && (!current || dirty & /*$$scope*/
        536870912)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[29],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[29]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[29],
              dirty,
              null
            ),
            null
          );
        }
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(default_slot, local);
      current = true;
    },
    o(local) {
      transition_out(default_slot, local);
      current = false;
    },
    d(detaching) {
      if (default_slot)
        default_slot.d(detaching);
    }
  };
}
function create_if_block_1(ctx) {
  let iconbutton;
  let current;
  iconbutton = new IconButton({
    props: {
      Icon: Upload$1,
      label: (
        /*i18n*/
        ctx[9]("common.upload")
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(iconbutton.$$.fragment);
    },
    l(nodes) {
      claim_component(iconbutton.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(iconbutton, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const iconbutton_changes = {};
      if (dirty & /*i18n*/
      512)
        iconbutton_changes.label = /*i18n*/
        ctx2[9]("common.upload");
      if (dirty & /*$$scope, file_types, file_count, max_file_size, root, stream_handler, upload, dragging, uploading*/
      536894642) {
        iconbutton_changes.$$scope = { dirty, ctx: ctx2 };
      }
      iconbutton.$set(iconbutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(iconbutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(iconbutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(iconbutton, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let upload_1;
  let updating_dragging;
  let updating_uploading;
  let current;
  function upload_1_dragging_binding(value) {
    ctx[19](value);
  }
  function upload_1_uploading_binding(value) {
    ctx[20](value);
  }
  let upload_1_props = {
    icon_upload: true,
    filetype: (
      /*file_types*/
      ctx[5]
    ),
    file_count: (
      /*file_count*/
      ctx[4]
    ),
    max_file_size: (
      /*max_file_size*/
      ctx[10]
    ),
    root: (
      /*root*/
      ctx[7]
    ),
    stream_handler: (
      /*stream_handler*/
      ctx[12]
    ),
    upload: (
      /*upload*/
      ctx[11]
    )
  };
  if (
    /*dragging*/
    ctx[14] !== void 0
  ) {
    upload_1_props.dragging = /*dragging*/
    ctx[14];
  }
  if (
    /*uploading*/
    ctx[1] !== void 0
  ) {
    upload_1_props.uploading = /*uploading*/
    ctx[1];
  }
  upload_1 = new Upload({ props: upload_1_props });
  binding_callbacks.push(() => bind(upload_1, "dragging", upload_1_dragging_binding));
  binding_callbacks.push(() => bind(upload_1, "uploading", upload_1_uploading_binding));
  upload_1.$on(
    "load",
    /*handle_upload*/
    ctx[15]
  );
  upload_1.$on(
    "error",
    /*error_handler*/
    ctx[21]
  );
  return {
    c() {
      create_component(upload_1.$$.fragment);
    },
    l(nodes) {
      claim_component(upload_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(upload_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const upload_1_changes = {};
      if (dirty & /*file_types*/
      32)
        upload_1_changes.filetype = /*file_types*/
        ctx2[5];
      if (dirty & /*file_count*/
      16)
        upload_1_changes.file_count = /*file_count*/
        ctx2[4];
      if (dirty & /*max_file_size*/
      1024)
        upload_1_changes.max_file_size = /*max_file_size*/
        ctx2[10];
      if (dirty & /*root*/
      128)
        upload_1_changes.root = /*root*/
        ctx2[7];
      if (dirty & /*stream_handler*/
      4096)
        upload_1_changes.stream_handler = /*stream_handler*/
        ctx2[12];
      if (dirty & /*upload*/
      2048)
        upload_1_changes.upload = /*upload*/
        ctx2[11];
      if (!updating_dragging && dirty & /*dragging*/
      16384) {
        updating_dragging = true;
        upload_1_changes.dragging = /*dragging*/
        ctx2[14];
        add_flush_callback(() => updating_dragging = false);
      }
      if (!updating_uploading && dirty & /*uploading*/
      2) {
        updating_uploading = true;
        upload_1_changes.uploading = /*uploading*/
        ctx2[1];
        add_flush_callback(() => updating_uploading = false);
      }
      upload_1.$set(upload_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(upload_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(upload_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(upload_1, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let show_if = !/*file_count*/
  (ctx[4] === "single" && (Array.isArray(
    /*value*/
    ctx[0]
  ) ? (
    /*value*/
    ctx[0].length > 0
  ) : (
    /*value*/
    ctx[0] !== null
  )));
  let t;
  let iconbutton;
  let current;
  let if_block = show_if && create_if_block_1(ctx);
  iconbutton = new IconButton({
    props: {
      Icon: Clear,
      label: (
        /*i18n*/
        ctx[9]("common.clear")
      )
    }
  });
  iconbutton.$on(
    "click",
    /*click_handler*/
    ctx[22]
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      t = space();
      create_component(iconbutton.$$.fragment);
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      t = claim_space(nodes);
      claim_component(iconbutton.$$.fragment, nodes);
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(iconbutton, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (dirty & /*file_count, value*/
      17)
        show_if = !/*file_count*/
        (ctx2[4] === "single" && (Array.isArray(
          /*value*/
          ctx2[0]
        ) ? (
          /*value*/
          ctx2[0].length > 0
        ) : (
          /*value*/
          ctx2[0] !== null
        )));
      if (show_if) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*file_count, value*/
          17) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block_1(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(t.parentNode, t);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      const iconbutton_changes = {};
      if (dirty & /*i18n*/
      512)
        iconbutton_changes.label = /*i18n*/
        ctx2[9]("common.clear");
      iconbutton.$set(iconbutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(iconbutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(iconbutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      if (if_block)
        if_block.d(detaching);
      destroy_component(iconbutton, detaching);
    }
  };
}
function create_fragment(ctx) {
  let blocklabel;
  let t;
  let show_if;
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  blocklabel = new BlockLabel({
    props: {
      show_label: (
        /*show_label*/
        ctx[3]
      ),
      Icon: File$1,
      float: !/*value*/
      ctx[0],
      label: (
        /*label*/
        ctx[2] || "File"
      )
    }
  });
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (dirty & /*value*/
    1)
      show_if = null;
    if (show_if == null)
      show_if = !!/*value*/
      (ctx2[0] && (Array.isArray(
        /*value*/
        ctx2[0]
      ) ? (
        /*value*/
        ctx2[0].length > 0
      ) : true));
    if (show_if)
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx, -1);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(blocklabel.$$.fragment);
      t = space();
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
      t = claim_space(nodes);
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      insert_hydration(target, t, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const blocklabel_changes = {};
      if (dirty & /*show_label*/
      8)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[3];
      if (dirty & /*value*/
      1)
        blocklabel_changes.float = !/*value*/
        ctx2[0];
      if (dirty & /*label*/
      4)
        blocklabel_changes.label = /*label*/
        ctx2[2] || "File";
      blocklabel.$set(blocklabel_changes);
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2, dirty);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        } else {
          if_block.p(ctx2, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(if_block_anchor.parentNode, if_block_anchor);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(blocklabel.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block_anchor);
      }
      destroy_component(blocklabel, detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  let { value } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { file_count = "single" } = $$props;
  let { file_types = null } = $$props;
  let { selectable = false } = $$props;
  let { root } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  let { max_file_size = null } = $$props;
  let { upload } = $$props;
  let { stream_handler } = $$props;
  let { uploading = false } = $$props;
  let { allow_reordering = false } = $$props;
  async function handle_upload({ detail }) {
    if (Array.isArray(value)) {
      $$invalidate(0, value = [...value, ...Array.isArray(detail) ? detail : [detail]]);
    } else if (value) {
      $$invalidate(0, value = [value, ...Array.isArray(detail) ? detail : [detail]]);
    } else {
      $$invalidate(0, value = detail);
    }
    await tick();
    dispatch("change", value);
    dispatch("upload", detail);
  }
  function handle_clear() {
    $$invalidate(0, value = null);
    dispatch("change", null);
    dispatch("clear");
  }
  const dispatch = createEventDispatcher();
  let dragging = false;
  function upload_1_dragging_binding(value2) {
    dragging = value2;
    $$invalidate(14, dragging);
  }
  function upload_1_uploading_binding(value2) {
    uploading = value2;
    $$invalidate(1, uploading);
  }
  function error_handler(event) {
    bubble.call(this, $$self, event);
  }
  const click_handler = (event) => {
    dispatch("clear");
    event.stopPropagation();
    handle_clear();
  };
  function select_handler(event) {
    bubble.call(this, $$self, event);
  }
  function change_handler(event) {
    bubble.call(this, $$self, event);
  }
  function delete_handler(event) {
    bubble.call(this, $$self, event);
  }
  function upload_1_dragging_binding_1(value2) {
    dragging = value2;
    $$invalidate(14, dragging);
  }
  function upload_1_uploading_binding_1(value2) {
    uploading = value2;
    $$invalidate(1, uploading);
  }
  function error_handler_1(event) {
    bubble.call(this, $$self, event);
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("label" in $$props2)
      $$invalidate(2, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(3, show_label = $$props2.show_label);
    if ("file_count" in $$props2)
      $$invalidate(4, file_count = $$props2.file_count);
    if ("file_types" in $$props2)
      $$invalidate(5, file_types = $$props2.file_types);
    if ("selectable" in $$props2)
      $$invalidate(6, selectable = $$props2.selectable);
    if ("root" in $$props2)
      $$invalidate(7, root = $$props2.root);
    if ("height" in $$props2)
      $$invalidate(8, height = $$props2.height);
    if ("i18n" in $$props2)
      $$invalidate(9, i18n = $$props2.i18n);
    if ("max_file_size" in $$props2)
      $$invalidate(10, max_file_size = $$props2.max_file_size);
    if ("upload" in $$props2)
      $$invalidate(11, upload = $$props2.upload);
    if ("stream_handler" in $$props2)
      $$invalidate(12, stream_handler = $$props2.stream_handler);
    if ("uploading" in $$props2)
      $$invalidate(1, uploading = $$props2.uploading);
    if ("allow_reordering" in $$props2)
      $$invalidate(13, allow_reordering = $$props2.allow_reordering);
    if ("$$scope" in $$props2)
      $$invalidate(29, $$scope = $$props2.$$scope);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*dragging*/
    16384) {
      dispatch("drag", dragging);
    }
  };
  return [
    value,
    uploading,
    label,
    show_label,
    file_count,
    file_types,
    selectable,
    root,
    height,
    i18n,
    max_file_size,
    upload,
    stream_handler,
    allow_reordering,
    dragging,
    handle_upload,
    handle_clear,
    dispatch,
    slots,
    upload_1_dragging_binding,
    upload_1_uploading_binding,
    error_handler,
    click_handler,
    select_handler,
    change_handler,
    delete_handler,
    upload_1_dragging_binding_1,
    upload_1_uploading_binding_1,
    error_handler_1,
    $$scope
  ];
}
class FileUpload extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      value: 0,
      label: 2,
      show_label: 3,
      file_count: 4,
      file_types: 5,
      selectable: 6,
      root: 7,
      height: 8,
      i18n: 9,
      max_file_size: 10,
      upload: 11,
      stream_handler: 12,
      uploading: 1,
      allow_reordering: 13
    });
  }
}
const BaseFileUpload = FileUpload;
export {
  BaseFileUpload as B,
  File as F,
  FilePreview$1 as a
};
