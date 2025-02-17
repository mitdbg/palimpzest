import { SvelteComponent, init, safe_not_equal, space, element, claim_space, claim_element, children, detach, attr, toggle_class, insert_hydration, listen, prevent_default, group_outros, transition_out, check_outros, transition_in, run_all, createEventDispatcher, set_input_value, stop_propagation, action_destroyer, bubble, binding_callbacks, text, claim_text, set_data, noop, create_component, claim_component, mount_component, destroy_component, HtmlTagHydration, empty, claim_html_tag, create_slot, add_render_callback, set_style, append_hydration, add_iframe_resize_listener, resize_observer_content_box, update_slot_base, get_all_dirty_from_scope, get_slot_changes, onMount, ensure_array_like, update_keyed_each, outro_and_destroy_block, tick, ResizeObserverSingleton, svg_element, claim_svg_element, is_function, onDestroy, bind, add_flush_callback, afterUpdate, globals, get_svelte_dataset, flush, assign, get_spread_update, get_spread_object } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { M as MarkdownCode, G as Copy, F as Check, B as Block, S as Static } from "./2.8WKXZUMv.js";
import { d as dequal } from "./index.DTU9giFV.js";
import { U as Upload } from "./Upload.C9nHsuEq.js";
/* empty css                                                    */
import { M as Maximize, a as Minimize } from "./Minimize.BNzzPy3I.js";
import { d as dsvFormat } from "./dsv.CKxpOsHB.js";
import { default as default2 } from "./Example.Cg4HFUYK.js";
function create_if_block_2$4(ctx) {
  let input;
  let mounted;
  let dispose;
  return {
    c() {
      input = element("input");
      this.h();
    },
    l(nodes) {
      input = claim_element(nodes, "INPUT", { role: true, tabindex: true, class: true });
      this.h();
    },
    h() {
      attr(input, "role", "textbox");
      attr(input, "tabindex", "-1");
      attr(input, "class", "svelte-1e50zov");
      toggle_class(
        input,
        "header",
        /*header*/
        ctx[5]
      );
    },
    m(target, anchor) {
      insert_hydration(target, input, anchor);
      ctx[21](input);
      set_input_value(
        input,
        /*_value*/
        ctx[11]
      );
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler*/
            ctx[22]
          ),
          listen(
            input,
            "blur",
            /*handle_blur*/
            ctx[13]
          ),
          listen(input, "mousedown", stop_propagation(
            /*mousedown_handler*/
            ctx[18]
          )),
          listen(input, "mouseup", stop_propagation(
            /*mouseup_handler*/
            ctx[19]
          )),
          listen(input, "click", stop_propagation(
            /*click_handler*/
            ctx[20]
          )),
          action_destroyer(
            /*use_focus*/
            ctx[12].call(null, input)
          ),
          listen(
            input,
            "keydown",
            /*handle_keydown*/
            ctx[14]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty & /*_value*/
      2048 && input.value !== /*_value*/
      ctx2[11]) {
        set_input_value(
          input,
          /*_value*/
          ctx2[11]
        );
      }
      if (dirty & /*header*/
      32) {
        toggle_class(
          input,
          "header",
          /*header*/
          ctx2[5]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(input);
      }
      ctx[21](null);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_else_block$1(ctx) {
  let t_value = (
    /*editable*/
    (ctx[9] ? (
      /*value*/
      ctx[0]
    ) : (
      /*display_value*/
      ctx[3] || /*value*/
      ctx[0]
    )) + ""
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
      if (dirty & /*editable, value, display_value*/
      521 && t_value !== (t_value = /*editable*/
      (ctx2[9] ? (
        /*value*/
        ctx2[0]
      ) : (
        /*display_value*/
        ctx2[3] || /*value*/
        ctx2[0]
      )) + ""))
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
function create_if_block_1$4(ctx) {
  let markdowncode;
  let current;
  markdowncode = new MarkdownCode({
    props: {
      message: (
        /*value*/
        ctx[0].toLocaleString()
      ),
      latex_delimiters: (
        /*latex_delimiters*/
        ctx[7]
      ),
      line_breaks: (
        /*line_breaks*/
        ctx[8]
      ),
      chatbot: false,
      root: (
        /*root*/
        ctx[10]
      )
    }
  });
  return {
    c() {
      create_component(markdowncode.$$.fragment);
    },
    l(nodes) {
      claim_component(markdowncode.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(markdowncode, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const markdowncode_changes = {};
      if (dirty & /*value*/
      1)
        markdowncode_changes.message = /*value*/
        ctx2[0].toLocaleString();
      if (dirty & /*latex_delimiters*/
      128)
        markdowncode_changes.latex_delimiters = /*latex_delimiters*/
        ctx2[7];
      if (dirty & /*line_breaks*/
      256)
        markdowncode_changes.line_breaks = /*line_breaks*/
        ctx2[8];
      if (dirty & /*root*/
      1024)
        markdowncode_changes.root = /*root*/
        ctx2[10];
      markdowncode.$set(markdowncode_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(markdowncode.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(markdowncode.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(markdowncode, detaching);
    }
  };
}
function create_if_block$5(ctx) {
  let html_tag;
  let html_anchor;
  return {
    c() {
      html_tag = new HtmlTagHydration(false);
      html_anchor = empty();
      this.h();
    },
    l(nodes) {
      html_tag = claim_html_tag(nodes, false);
      html_anchor = empty();
      this.h();
    },
    h() {
      html_tag.a = html_anchor;
    },
    m(target, anchor) {
      html_tag.m(
        /*value*/
        ctx[0],
        target,
        anchor
      );
      insert_hydration(target, html_anchor, anchor);
    },
    p(ctx2, dirty) {
      if (dirty & /*value*/
      1)
        html_tag.p(
          /*value*/
          ctx2[0]
        );
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(html_anchor);
        html_tag.d();
      }
    }
  };
}
function create_fragment$6(ctx) {
  let t;
  let span;
  let current_block_type_index;
  let if_block1;
  let current;
  let mounted;
  let dispose;
  let if_block0 = (
    /*edit*/
    ctx[2] && create_if_block_2$4(ctx)
  );
  const if_block_creators = [create_if_block$5, create_if_block_1$4, create_else_block$1];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*datatype*/
      ctx2[6] === "html"
    )
      return 0;
    if (
      /*datatype*/
      ctx2[6] === "markdown"
    )
      return 1;
    return 2;
  }
  current_block_type_index = select_block_type(ctx);
  if_block1 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      if (if_block0)
        if_block0.c();
      t = space();
      span = element("span");
      if_block1.c();
      this.h();
    },
    l(nodes) {
      if (if_block0)
        if_block0.l(nodes);
      t = claim_space(nodes);
      span = claim_element(nodes, "SPAN", {
        tabindex: true,
        role: true,
        style: true,
        class: true,
        placeholder: true
      });
      var span_nodes = children(span);
      if_block1.l(span_nodes);
      span_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(span, "tabindex", "-1");
      attr(span, "role", "button");
      attr(
        span,
        "style",
        /*styling*/
        ctx[4]
      );
      attr(span, "class", "table-cell-text svelte-1e50zov");
      attr(span, "placeholder", " ");
      toggle_class(
        span,
        "edit",
        /*edit*/
        ctx[2]
      );
    },
    m(target, anchor) {
      if (if_block0)
        if_block0.m(target, anchor);
      insert_hydration(target, t, anchor);
      insert_hydration(target, span, anchor);
      if_blocks[current_block_type_index].m(span, null);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            span,
            "dblclick",
            /*dblclick_handler*/
            ctx[16]
          ),
          listen(span, "focus", prevent_default(
            /*focus_handler*/
            ctx[17]
          ))
        ];
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (
        /*edit*/
        ctx2[2]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
        } else {
          if_block0 = create_if_block_2$4(ctx2);
          if_block0.c();
          if_block0.m(t.parentNode, t);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block1 = if_blocks[current_block_type_index];
        if (!if_block1) {
          if_block1 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block1.c();
        } else {
          if_block1.p(ctx2, dirty);
        }
        transition_in(if_block1, 1);
        if_block1.m(span, null);
      }
      if (!current || dirty & /*styling*/
      16) {
        attr(
          span,
          "style",
          /*styling*/
          ctx2[4]
        );
      }
      if (!current || dirty & /*edit*/
      4) {
        toggle_class(
          span,
          "edit",
          /*edit*/
          ctx2[2]
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
        detach(t);
        detach(span);
      }
      if (if_block0)
        if_block0.d(detaching);
      if_blocks[current_block_type_index].d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$6($$self, $$props, $$invalidate) {
  let _value;
  let { edit } = $$props;
  let { value = "" } = $$props;
  let { display_value = null } = $$props;
  let { styling = "" } = $$props;
  let { header = false } = $$props;
  let { datatype = "str" } = $$props;
  let { latex_delimiters } = $$props;
  let { clear_on_focus = false } = $$props;
  let { line_breaks = true } = $$props;
  let { editable = true } = $$props;
  let { root } = $$props;
  const dispatch = createEventDispatcher();
  let { el } = $$props;
  function use_focus(node) {
    if (clear_on_focus) {
      $$invalidate(11, _value = "");
    }
    requestAnimationFrame(() => {
      node.focus();
    });
    return {};
  }
  function handle_blur({ currentTarget }) {
    $$invalidate(0, value = currentTarget.value);
    dispatch("blur");
  }
  function handle_keydown(event) {
    if (event.key === "Enter") {
      $$invalidate(0, value = _value);
      dispatch("blur");
    }
    dispatch("keydown", event);
  }
  function dblclick_handler(event) {
    bubble.call(this, $$self, event);
  }
  function focus_handler(event) {
    bubble.call(this, $$self, event);
  }
  function mousedown_handler2(event) {
    bubble.call(this, $$self, event);
  }
  function mouseup_handler(event) {
    bubble.call(this, $$self, event);
  }
  function click_handler(event) {
    bubble.call(this, $$self, event);
  }
  function input_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      el = $$value;
      $$invalidate(1, el);
    });
  }
  function input_input_handler() {
    _value = this.value;
    $$invalidate(11, _value), $$invalidate(0, value);
  }
  $$self.$$set = ($$props2) => {
    if ("edit" in $$props2)
      $$invalidate(2, edit = $$props2.edit);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("display_value" in $$props2)
      $$invalidate(3, display_value = $$props2.display_value);
    if ("styling" in $$props2)
      $$invalidate(4, styling = $$props2.styling);
    if ("header" in $$props2)
      $$invalidate(5, header = $$props2.header);
    if ("datatype" in $$props2)
      $$invalidate(6, datatype = $$props2.datatype);
    if ("latex_delimiters" in $$props2)
      $$invalidate(7, latex_delimiters = $$props2.latex_delimiters);
    if ("clear_on_focus" in $$props2)
      $$invalidate(15, clear_on_focus = $$props2.clear_on_focus);
    if ("line_breaks" in $$props2)
      $$invalidate(8, line_breaks = $$props2.line_breaks);
    if ("editable" in $$props2)
      $$invalidate(9, editable = $$props2.editable);
    if ("root" in $$props2)
      $$invalidate(10, root = $$props2.root);
    if ("el" in $$props2)
      $$invalidate(1, el = $$props2.el);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value*/
    1) {
      $$invalidate(11, _value = value);
    }
  };
  return [
    value,
    el,
    edit,
    display_value,
    styling,
    header,
    datatype,
    latex_delimiters,
    line_breaks,
    editable,
    root,
    _value,
    use_focus,
    handle_blur,
    handle_keydown,
    clear_on_focus,
    dblclick_handler,
    focus_handler,
    mousedown_handler2,
    mouseup_handler,
    click_handler,
    input_binding,
    input_input_handler
  ];
}
class EditableCell extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$6, create_fragment$6, safe_not_equal, {
      edit: 2,
      value: 0,
      display_value: 3,
      styling: 4,
      header: 5,
      datatype: 6,
      latex_delimiters: 7,
      clear_on_focus: 15,
      line_breaks: 8,
      editable: 9,
      root: 10,
      el: 1
    });
  }
}
const get_tfoot_slot_changes = (dirty) => ({});
const get_tfoot_slot_context = (ctx) => ({});
function get_each_context$1(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[36] = list[i];
  return child_ctx;
}
const get_tbody_slot_changes = (dirty) => ({
  item: dirty[0] & /*visible*/
  256,
  index: dirty[0] & /*visible*/
  256
});
const get_tbody_slot_context = (ctx) => ({
  item: (
    /*item*/
    ctx[36].data
  ),
  index: (
    /*item*/
    ctx[36].index
  )
});
const get_thead_slot_changes = (dirty) => ({});
const get_thead_slot_context = (ctx) => ({});
function create_if_block$4(ctx) {
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let each_1_anchor;
  let current;
  let each_value = ensure_array_like(
    /*visible*/
    ctx[8]
  );
  const get_key = (ctx2) => (
    /*item*/
    ctx2[36].data[0].id
  );
  for (let i = 0; i < each_value.length; i += 1) {
    let child_ctx = get_each_context$1(ctx, each_value, i);
    let key = get_key(child_ctx);
    each_1_lookup.set(key, each_blocks[i] = create_each_block$1(key, child_ctx));
  }
  return {
    c() {
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      each_1_anchor = empty();
    },
    l(nodes) {
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(nodes);
      }
      each_1_anchor = empty();
    },
    m(target, anchor) {
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(target, anchor);
        }
      }
      insert_hydration(target, each_1_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*$$scope, visible*/
      1048832) {
        each_value = ensure_array_like(
          /*visible*/
          ctx2[8]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value, each_1_lookup, each_1_anchor.parentNode, outro_and_destroy_block, create_each_block$1, each_1_anchor, get_each_context$1);
        check_outros();
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
        detach(each_1_anchor);
      }
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d(detaching);
      }
    }
  };
}
function fallback_block(ctx) {
  let t;
  return {
    c() {
      t = text("Missing Table Row\n					");
    },
    l(nodes) {
      t = claim_text(nodes, "Missing Table Row\n					");
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_each_block$1(key_1, ctx) {
  let first;
  let current;
  const tbody_slot_template = (
    /*#slots*/
    ctx[21].tbody
  );
  const tbody_slot = create_slot(
    tbody_slot_template,
    ctx,
    /*$$scope*/
    ctx[20],
    get_tbody_slot_context
  );
  const tbody_slot_or_fallback = tbody_slot || fallback_block();
  return {
    key: key_1,
    first: null,
    c() {
      first = empty();
      if (tbody_slot_or_fallback)
        tbody_slot_or_fallback.c();
      this.h();
    },
    l(nodes) {
      first = empty();
      if (tbody_slot_or_fallback)
        tbody_slot_or_fallback.l(nodes);
      this.h();
    },
    h() {
      this.first = first;
    },
    m(target, anchor) {
      insert_hydration(target, first, anchor);
      if (tbody_slot_or_fallback) {
        tbody_slot_or_fallback.m(target, anchor);
      }
      current = true;
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (tbody_slot) {
        if (tbody_slot.p && (!current || dirty[0] & /*$$scope, visible*/
        1048832)) {
          update_slot_base(
            tbody_slot,
            tbody_slot_template,
            ctx,
            /*$$scope*/
            ctx[20],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx[20]
            ) : get_slot_changes(
              tbody_slot_template,
              /*$$scope*/
              ctx[20],
              dirty,
              get_tbody_slot_changes
            ),
            get_tbody_slot_context
          );
        }
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(tbody_slot_or_fallback, local);
      current = true;
    },
    o(local) {
      transition_out(tbody_slot_or_fallback, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(first);
      }
      if (tbody_slot_or_fallback)
        tbody_slot_or_fallback.d(detaching);
    }
  };
}
function create_fragment$5(ctx) {
  let svelte_virtual_table_viewport;
  let table;
  let thead;
  let thead_resize_listener;
  let t0;
  let tbody;
  let t1;
  let tfoot;
  let tfoot_resize_listener;
  let table_resize_listener;
  let current;
  let mounted;
  let dispose;
  const thead_slot_template = (
    /*#slots*/
    ctx[21].thead
  );
  const thead_slot = create_slot(
    thead_slot_template,
    ctx,
    /*$$scope*/
    ctx[20],
    get_thead_slot_context
  );
  let if_block = (
    /*visible*/
    ctx[8].length && /*visible*/
    ctx[8][0].data.length && create_if_block$4(ctx)
  );
  const tfoot_slot_template = (
    /*#slots*/
    ctx[21].tfoot
  );
  const tfoot_slot = create_slot(
    tfoot_slot_template,
    ctx,
    /*$$scope*/
    ctx[20],
    get_tfoot_slot_context
  );
  return {
    c() {
      svelte_virtual_table_viewport = element("svelte-virtual-table-viewport");
      table = element("table");
      thead = element("thead");
      if (thead_slot)
        thead_slot.c();
      t0 = space();
      tbody = element("tbody");
      if (if_block)
        if_block.c();
      t1 = space();
      tfoot = element("tfoot");
      if (tfoot_slot)
        tfoot_slot.c();
      this.h();
    },
    l(nodes) {
      svelte_virtual_table_viewport = claim_element(nodes, "SVELTE-VIRTUAL-TABLE-VIEWPORT", {});
      var svelte_virtual_table_viewport_nodes = children(svelte_virtual_table_viewport);
      table = claim_element(svelte_virtual_table_viewport_nodes, "TABLE", { class: true, style: true });
      var table_nodes = children(table);
      thead = claim_element(table_nodes, "THEAD", { class: true });
      var thead_nodes = children(thead);
      if (thead_slot)
        thead_slot.l(thead_nodes);
      thead_nodes.forEach(detach);
      t0 = claim_space(table_nodes);
      tbody = claim_element(table_nodes, "TBODY", { class: true });
      var tbody_nodes = children(tbody);
      if (if_block)
        if_block.l(tbody_nodes);
      tbody_nodes.forEach(detach);
      t1 = claim_space(table_nodes);
      tfoot = claim_element(table_nodes, "TFOOT", { class: true });
      var tfoot_nodes = children(tfoot);
      if (tfoot_slot)
        tfoot_slot.l(tfoot_nodes);
      tfoot_nodes.forEach(detach);
      table_nodes.forEach(detach);
      svelte_virtual_table_viewport_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(thead, "class", "thead svelte-1xyl3gk");
      add_render_callback(() => (
        /*thead_elementresize_handler*/
        ctx[22].call(thead)
      ));
      attr(tbody, "class", "tbody svelte-1xyl3gk");
      attr(tfoot, "class", "tfoot svelte-1xyl3gk");
      add_render_callback(() => (
        /*tfoot_elementresize_handler*/
        ctx[24].call(tfoot)
      ));
      attr(table, "class", "table svelte-1xyl3gk");
      set_style(table, "height", height);
      set_style(
        table,
        "--bw-svt-p-top",
        /*top*/
        ctx[6] + "px"
      );
      set_style(
        table,
        "--bw-svt-p-bottom",
        /*bottom*/
        ctx[2] + "px"
      );
      set_style(
        table,
        "--bw-svt-head-height",
        /*head_height*/
        ctx[4] + "px"
      );
      set_style(
        table,
        "--bw-svt-foot-height",
        /*foot_height*/
        ctx[5] + "px"
      );
      set_style(
        table,
        "--bw-svt-avg-row-height",
        /*average_height*/
        ctx[0] + "px"
      );
    },
    m(target, anchor) {
      insert_hydration(target, svelte_virtual_table_viewport, anchor);
      append_hydration(svelte_virtual_table_viewport, table);
      append_hydration(table, thead);
      if (thead_slot) {
        thead_slot.m(thead, null);
      }
      thead_resize_listener = add_iframe_resize_listener(
        thead,
        /*thead_elementresize_handler*/
        ctx[22].bind(thead)
      );
      append_hydration(table, t0);
      append_hydration(table, tbody);
      if (if_block)
        if_block.m(tbody, null);
      ctx[23](tbody);
      append_hydration(table, t1);
      append_hydration(table, tfoot);
      if (tfoot_slot) {
        tfoot_slot.m(tfoot, null);
      }
      tfoot_resize_listener = add_iframe_resize_listener(
        tfoot,
        /*tfoot_elementresize_handler*/
        ctx[24].bind(tfoot)
      );
      ctx[25](table);
      table_resize_listener = resize_observer_content_box.observe(
        table,
        /*table_elementresizecontentbox_handler*/
        ctx[26].bind(table)
      );
      current = true;
      if (!mounted) {
        dispose = listen(
          table,
          "scroll",
          /*handle_scroll*/
          ctx[9]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (thead_slot) {
        if (thead_slot.p && (!current || dirty[0] & /*$$scope*/
        1048576)) {
          update_slot_base(
            thead_slot,
            thead_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[20],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[20]
            ) : get_slot_changes(
              thead_slot_template,
              /*$$scope*/
              ctx2[20],
              dirty,
              get_thead_slot_changes
            ),
            get_thead_slot_context
          );
        }
      }
      if (
        /*visible*/
        ctx2[8].length && /*visible*/
        ctx2[8][0].data.length
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty[0] & /*visible*/
          256) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block$4(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(tbody, null);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (tfoot_slot) {
        if (tfoot_slot.p && (!current || dirty[0] & /*$$scope*/
        1048576)) {
          update_slot_base(
            tfoot_slot,
            tfoot_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[20],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[20]
            ) : get_slot_changes(
              tfoot_slot_template,
              /*$$scope*/
              ctx2[20],
              dirty,
              get_tfoot_slot_changes
            ),
            get_tfoot_slot_context
          );
        }
      }
      if (!current || dirty[0] & /*top*/
      64) {
        set_style(
          table,
          "--bw-svt-p-top",
          /*top*/
          ctx2[6] + "px"
        );
      }
      if (!current || dirty[0] & /*bottom*/
      4) {
        set_style(
          table,
          "--bw-svt-p-bottom",
          /*bottom*/
          ctx2[2] + "px"
        );
      }
      if (!current || dirty[0] & /*head_height*/
      16) {
        set_style(
          table,
          "--bw-svt-head-height",
          /*head_height*/
          ctx2[4] + "px"
        );
      }
      if (!current || dirty[0] & /*foot_height*/
      32) {
        set_style(
          table,
          "--bw-svt-foot-height",
          /*foot_height*/
          ctx2[5] + "px"
        );
      }
      if (!current || dirty[0] & /*average_height*/
      1) {
        set_style(
          table,
          "--bw-svt-avg-row-height",
          /*average_height*/
          ctx2[0] + "px"
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(thead_slot, local);
      transition_in(if_block);
      transition_in(tfoot_slot, local);
      current = true;
    },
    o(local) {
      transition_out(thead_slot, local);
      transition_out(if_block);
      transition_out(tfoot_slot, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(svelte_virtual_table_viewport);
      }
      if (thead_slot)
        thead_slot.d(detaching);
      thead_resize_listener();
      if (if_block)
        if_block.d();
      ctx[23](null);
      if (tfoot_slot)
        tfoot_slot.d(detaching);
      tfoot_resize_listener();
      ctx[25](null);
      table_resize_listener();
      mounted = false;
      dispose();
    }
  };
}
let height = "100%";
function get_computed_px_amount(elem, property) {
  if (!elem) {
    return 0;
  }
  const compStyle = getComputedStyle(elem);
  let x = parseInt(compStyle.getPropertyValue(property));
  return x;
}
function instance$5($$self, $$props, $$invalidate) {
  let sortedItems;
  let { $$slots: slots = {}, $$scope } = $$props;
  let { items = [] } = $$props;
  let { max_height } = $$props;
  let { actual_height } = $$props;
  let { table_scrollbar_width } = $$props;
  let { start = 0 } = $$props;
  let { end = 20 } = $$props;
  let { selected } = $$props;
  let average_height = 30;
  let bottom = 0;
  let contents;
  let head_height = 0;
  let foot_height = 0;
  let height_map = [];
  let mounted;
  let rows;
  let top = 0;
  let viewport;
  let viewport_height = 200;
  let visible = [];
  let viewport_box;
  const is_browser = typeof window !== "undefined";
  const raf = is_browser ? window.requestAnimationFrame : (cb) => cb();
  let content_height = 0;
  async function refresh_height_map(_items) {
    if (viewport_height === 0) {
      return;
    }
    const { scrollTop } = viewport;
    $$invalidate(13, table_scrollbar_width = viewport.offsetWidth - viewport.clientWidth);
    content_height = top - (scrollTop - head_height);
    let i = start;
    while (content_height < max_height && i < _items.length) {
      let row = rows[i - start];
      if (!row) {
        $$invalidate(11, end = i + 1);
        await tick();
        row = rows[i - start];
      }
      let _h = row == null ? void 0 : row.getBoundingClientRect().height;
      if (!_h) {
        _h = average_height;
      }
      const row_height = height_map[i] = _h;
      content_height += row_height;
      i += 1;
    }
    $$invalidate(11, end = i);
    const remaining = _items.length - end;
    const scrollbar_height = viewport.offsetHeight - viewport.clientHeight;
    if (scrollbar_height > 0) {
      content_height += scrollbar_height;
    }
    let filtered_height_map = height_map.filter((v) => typeof v === "number");
    $$invalidate(0, average_height = filtered_height_map.reduce((a, b) => a + b, 0) / filtered_height_map.length);
    $$invalidate(2, bottom = remaining * average_height);
    height_map.length = _items.length;
    await tick();
    if (!max_height) {
      $$invalidate(12, actual_height = content_height + 1);
    } else if (content_height < max_height) {
      $$invalidate(12, actual_height = content_height + 2);
    } else {
      $$invalidate(12, actual_height = max_height);
    }
    await tick();
  }
  async function scroll_and_render(n) {
    raf(async () => {
      if (typeof n !== "number")
        return;
      const direction = typeof n !== "number" ? false : is_in_view(n);
      if (direction === true) {
        return;
      }
      if (direction === "back") {
        await scroll_to_index(n, { behavior: "instant" });
      }
      if (direction === "forwards") {
        await scroll_to_index(n, { behavior: "instant" }, true);
      }
    });
  }
  function is_in_view(n) {
    const current = rows && rows[n - start];
    if (!current && n < start) {
      return "back";
    }
    if (!current && n >= end - 1) {
      return "forwards";
    }
    const { top: viewport_top } = viewport.getBoundingClientRect();
    const { top: top2, bottom: bottom2 } = current.getBoundingClientRect();
    if (top2 - viewport_top < 37) {
      return "back";
    }
    if (bottom2 - viewport_top > viewport_height) {
      return "forwards";
    }
    return true;
  }
  async function handle_scroll(e) {
    const scroll_top = viewport.scrollTop;
    rows = contents.children;
    const is_start_overflow = sortedItems.length < start;
    const row_top_border = get_computed_px_amount(rows[1], "border-top-width");
    const actual_border_collapsed_width = 0;
    if (is_start_overflow) {
      await scroll_to_index(sortedItems.length - 1, { behavior: "auto" });
    }
    let new_start = 0;
    for (let v = 0; v < rows.length; v += 1) {
      height_map[start + v] = rows[v].getBoundingClientRect().height;
    }
    let i = 0;
    let y = head_height + row_top_border / 2;
    let row_heights = [];
    while (i < sortedItems.length) {
      const row_height = height_map[i] || average_height;
      row_heights[i] = row_height;
      if (y + row_height + actual_border_collapsed_width > scroll_top) {
        new_start = i;
        $$invalidate(6, top = y - (head_height + row_top_border / 2));
        break;
      }
      y += row_height;
      i += 1;
    }
    new_start = Math.max(0, new_start);
    while (i < sortedItems.length) {
      const row_height = height_map[i] || average_height;
      y += row_height;
      i += 1;
      if (y > scroll_top + viewport_height) {
        break;
      }
    }
    $$invalidate(10, start = new_start);
    $$invalidate(11, end = i);
    const remaining = sortedItems.length - end;
    if (end === 0) {
      $$invalidate(11, end = 10);
    }
    $$invalidate(0, average_height = (y - head_height) / end);
    let remaining_height = remaining * average_height;
    while (i < sortedItems.length) {
      i += 1;
      height_map[i] = average_height;
    }
    $$invalidate(2, bottom = remaining_height);
    if (!isFinite(bottom)) {
      $$invalidate(2, bottom = 2e5);
    }
  }
  async function scroll_to_index(index, opts, align_end = false) {
    await tick();
    const _itemHeight = average_height;
    let distance = index * _itemHeight;
    if (align_end) {
      distance = distance - viewport_height + _itemHeight + head_height;
    }
    const scrollbar_height = viewport.offsetHeight - viewport.clientHeight;
    if (scrollbar_height > 0) {
      distance += scrollbar_height;
    }
    const _opts = {
      top: distance,
      behavior: "smooth",
      ...opts
    };
    viewport.scrollTo(_opts);
  }
  onMount(() => {
    rows = contents.children;
    $$invalidate(18, mounted = true);
    refresh_height_map(items);
  });
  function thead_elementresize_handler() {
    head_height = this.offsetHeight;
    $$invalidate(4, head_height);
  }
  function tbody_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      contents = $$value;
      $$invalidate(3, contents);
    });
  }
  function tfoot_elementresize_handler() {
    foot_height = this.offsetHeight;
    $$invalidate(5, foot_height);
  }
  function table_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      viewport = $$value;
      $$invalidate(7, viewport);
    });
  }
  function table_elementresizecontentbox_handler() {
    var _a;
    viewport_box = (_a = ResizeObserverSingleton.entries.get(this)) == null ? void 0 : _a.contentRect;
    $$invalidate(1, viewport_box);
  }
  $$self.$$set = ($$props2) => {
    if ("items" in $$props2)
      $$invalidate(14, items = $$props2.items);
    if ("max_height" in $$props2)
      $$invalidate(15, max_height = $$props2.max_height);
    if ("actual_height" in $$props2)
      $$invalidate(12, actual_height = $$props2.actual_height);
    if ("table_scrollbar_width" in $$props2)
      $$invalidate(13, table_scrollbar_width = $$props2.table_scrollbar_width);
    if ("start" in $$props2)
      $$invalidate(10, start = $$props2.start);
    if ("end" in $$props2)
      $$invalidate(11, end = $$props2.end);
    if ("selected" in $$props2)
      $$invalidate(16, selected = $$props2.selected);
    if ("$$scope" in $$props2)
      $$invalidate(20, $$scope = $$props2.$$scope);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*viewport_box*/
    2) {
      viewport_height = (viewport_box == null ? void 0 : viewport_box.height) || 200;
    }
    if ($$self.$$.dirty[0] & /*items*/
    16384) {
      $$invalidate(19, sortedItems = items);
    }
    if ($$self.$$.dirty[0] & /*mounted, sortedItems*/
    786432) {
      mounted && raf(() => refresh_height_map(sortedItems));
    }
    if ($$self.$$.dirty[0] & /*selected*/
    65536) {
      scroll_and_render(selected);
    }
    if ($$self.$$.dirty[0] & /*sortedItems, start, end, max_height, average_height*/
    560129) {
      $$invalidate(8, visible = is_browser ? sortedItems.slice(start, end).map((data, i) => {
        return { index: i + start, data };
      }) : sortedItems.slice(0, max_height / sortedItems.length * average_height + 1).map((data, i) => {
        return { index: i + start, data };
      }));
    }
  };
  return [
    average_height,
    viewport_box,
    bottom,
    contents,
    head_height,
    foot_height,
    top,
    viewport,
    visible,
    handle_scroll,
    start,
    end,
    actual_height,
    table_scrollbar_width,
    items,
    max_height,
    selected,
    scroll_to_index,
    mounted,
    sortedItems,
    $$scope,
    slots,
    thead_elementresize_handler,
    tbody_binding,
    tfoot_elementresize_handler,
    table_binding,
    table_elementresizecontentbox_handler
  ];
}
class VirtualTable extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance$5,
      create_fragment$5,
      safe_not_equal,
      {
        items: 14,
        max_height: 15,
        actual_height: 12,
        table_scrollbar_width: 13,
        start: 10,
        end: 11,
        selected: 16,
        scroll_to_index: 17
      },
      null,
      [-1, -1]
    );
  }
  get scroll_to_index() {
    return this.$$.ctx[17];
  }
}
function create_if_block_5$1(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "10");
      attr(rect, "y", "5");
      attr(rect, "width", "4");
      attr(rect, "height", "14");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(path, "d", "M7 8L17 16M17 8L7 16");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_if_block_4$1(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "5");
      attr(rect, "y", "10");
      attr(rect, "width", "14");
      attr(rect, "height", "4");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(path, "d", "M8 7L16 17M16 7L8 17");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_if_block_3$3(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        fill: true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "6");
      attr(rect, "y", "4");
      attr(rect, "width", "12");
      attr(rect, "height", "4");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(path, "d", "M12 12V19M8 16L12 19L16 16");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "fill", "none");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_if_block_2$3(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        fill: true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "6");
      attr(rect, "y", "16");
      attr(rect, "width", "12");
      attr(rect, "height", "4");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(path, "d", "M12 12V5M8 8L12 5L16 8");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "fill", "none");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_if_block_1$3(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true,
        fill: true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        fill: true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "16");
      attr(rect, "y", "6");
      attr(rect, "width", "4");
      attr(rect, "height", "12");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(rect, "fill", "none");
      attr(path, "d", "M12 12H5M8 8L5 12L8 16");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "fill", "none");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_if_block$3(ctx) {
  let svg;
  let rect;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", { viewBox: true, width: true, height: true });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true,
        fill: true
      });
      children(rect).forEach(detach);
      path = claim_svg_element(svg_nodes, "path", {
        d: true,
        stroke: true,
        "stroke-width": true,
        fill: true,
        "stroke-linecap": true
      });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "4");
      attr(rect, "y", "6");
      attr(rect, "width", "4");
      attr(rect, "height", "12");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(rect, "fill", "none");
      attr(path, "d", "M12 12H19M16 8L19 12L16 16");
      attr(path, "stroke", "currentColor");
      attr(path, "stroke-width", "2");
      attr(path, "fill", "none");
      attr(path, "stroke-linecap", "round");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, path);
    },
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
function create_fragment$4(ctx) {
  let if_block_anchor;
  function select_block_type(ctx2, dirty) {
    if (
      /*icon*/
      ctx2[0] == "add-column-right"
    )
      return create_if_block$3;
    if (
      /*icon*/
      ctx2[0] == "add-column-left"
    )
      return create_if_block_1$3;
    if (
      /*icon*/
      ctx2[0] == "add-row-above"
    )
      return create_if_block_2$3;
    if (
      /*icon*/
      ctx2[0] == "add-row-below"
    )
      return create_if_block_3$3;
    if (
      /*icon*/
      ctx2[0] == "delete-row"
    )
      return create_if_block_4$1;
    if (
      /*icon*/
      ctx2[0] == "delete-column"
    )
      return create_if_block_5$1;
  }
  let current_block_type = select_block_type(ctx);
  let if_block = current_block_type && current_block_type(ctx);
  return {
    c() {
      if (if_block)
        if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
    },
    p(ctx2, [dirty]) {
      if (current_block_type !== (current_block_type = select_block_type(ctx2))) {
        if (if_block)
          if_block.d(1);
        if_block = current_block_type && current_block_type(ctx2);
        if (if_block) {
          if_block.c();
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      }
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(if_block_anchor);
      }
      if (if_block) {
        if_block.d(detaching);
      }
    }
  };
}
function instance$4($$self, $$props, $$invalidate) {
  let { icon } = $$props;
  $$self.$$set = ($$props2) => {
    if ("icon" in $$props2)
      $$invalidate(0, icon = $$props2.icon);
  };
  return [icon];
}
class CellMenuIcons extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$4, create_fragment$4, safe_not_equal, { icon: 0 });
  }
}
function create_if_block_2$2(ctx) {
  let button0;
  let cellmenuicons0;
  let t0;
  let t1_value = (
    /*i18n*/
    ctx[8]("dataframe.add_row_above") + ""
  );
  let t1;
  let t2;
  let button1;
  let cellmenuicons1;
  let t3;
  let t4_value = (
    /*i18n*/
    ctx[8]("dataframe.add_row_below") + ""
  );
  let t4;
  let t5;
  let if_block_anchor;
  let current;
  let mounted;
  let dispose;
  cellmenuicons0 = new CellMenuIcons({ props: { icon: "add-row-above" } });
  cellmenuicons1 = new CellMenuIcons({ props: { icon: "add-row-below" } });
  let if_block = (
    /*can_delete_rows*/
    ctx[6] && create_if_block_3$2(ctx)
  );
  return {
    c() {
      button0 = element("button");
      create_component(cellmenuicons0.$$.fragment);
      t0 = space();
      t1 = text(t1_value);
      t2 = space();
      button1 = element("button");
      create_component(cellmenuicons1.$$.fragment);
      t3 = space();
      t4 = text(t4_value);
      t5 = space();
      if (if_block)
        if_block.c();
      if_block_anchor = empty();
      this.h();
    },
    l(nodes) {
      button0 = claim_element(nodes, "BUTTON", { class: true });
      var button0_nodes = children(button0);
      claim_component(cellmenuicons0.$$.fragment, button0_nodes);
      t0 = claim_space(button0_nodes);
      t1 = claim_text(button0_nodes, t1_value);
      button0_nodes.forEach(detach);
      t2 = claim_space(nodes);
      button1 = claim_element(nodes, "BUTTON", { class: true });
      var button1_nodes = children(button1);
      claim_component(cellmenuicons1.$$.fragment, button1_nodes);
      t3 = claim_space(button1_nodes);
      t4 = claim_text(button1_nodes, t4_value);
      button1_nodes.forEach(detach);
      t5 = claim_space(nodes);
      if (if_block)
        if_block.l(nodes);
      if_block_anchor = empty();
      this.h();
    },
    h() {
      attr(button0, "class", "svelte-9g877");
      attr(button1, "class", "svelte-9g877");
    },
    m(target, anchor) {
      insert_hydration(target, button0, anchor);
      mount_component(cellmenuicons0, button0, null);
      append_hydration(button0, t0);
      append_hydration(button0, t1);
      insert_hydration(target, t2, anchor);
      insert_hydration(target, button1, anchor);
      mount_component(cellmenuicons1, button1, null);
      append_hydration(button1, t3);
      append_hydration(button1, t4);
      insert_hydration(target, t5, anchor);
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            button0,
            "click",
            /*click_handler*/
            ctx[18]
          ),
          listen(
            button1,
            "click",
            /*click_handler_1*/
            ctx[19]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if ((!current || dirty & /*i18n*/
      256) && t1_value !== (t1_value = /*i18n*/
      ctx2[8]("dataframe.add_row_above") + ""))
        set_data(t1, t1_value);
      if ((!current || dirty & /*i18n*/
      256) && t4_value !== (t4_value = /*i18n*/
      ctx2[8]("dataframe.add_row_below") + ""))
        set_data(t4, t4_value);
      if (
        /*can_delete_rows*/
        ctx2[6]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*can_delete_rows*/
          64) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block_3$2(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenuicons0.$$.fragment, local);
      transition_in(cellmenuicons1.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(cellmenuicons0.$$.fragment, local);
      transition_out(cellmenuicons1.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button0);
        detach(t2);
        detach(button1);
        detach(t5);
        detach(if_block_anchor);
      }
      destroy_component(cellmenuicons0);
      destroy_component(cellmenuicons1);
      if (if_block)
        if_block.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_if_block_3$2(ctx) {
  let button;
  let cellmenuicons;
  let t0;
  let t1_value = (
    /*i18n*/
    ctx[8]("dataframe.delete_row") + ""
  );
  let t1;
  let current;
  let mounted;
  let dispose;
  cellmenuicons = new CellMenuIcons({ props: { icon: "delete-row" } });
  return {
    c() {
      button = element("button");
      create_component(cellmenuicons.$$.fragment);
      t0 = space();
      t1 = text(t1_value);
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      claim_component(cellmenuicons.$$.fragment, button_nodes);
      t0 = claim_space(button_nodes);
      t1 = claim_text(button_nodes, t1_value);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "delete svelte-9g877");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      mount_component(cellmenuicons, button, null);
      append_hydration(button, t0);
      append_hydration(button, t1);
      current = true;
      if (!mounted) {
        dispose = listen(button, "click", function() {
          if (is_function(
            /*on_delete_row*/
            ctx[4]
          ))
            ctx[4].apply(this, arguments);
        });
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if ((!current || dirty & /*i18n*/
      256) && t1_value !== (t1_value = /*i18n*/
      ctx[8]("dataframe.delete_row") + ""))
        set_data(t1, t1_value);
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenuicons.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(cellmenuicons.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      destroy_component(cellmenuicons);
      mounted = false;
      dispose();
    }
  };
}
function create_if_block$2(ctx) {
  let button0;
  let cellmenuicons0;
  let t0;
  let t1_value = (
    /*i18n*/
    ctx[8]("dataframe.add_column_left") + ""
  );
  let t1;
  let t2;
  let button1;
  let cellmenuicons1;
  let t3;
  let t4_value = (
    /*i18n*/
    ctx[8]("dataframe.add_column_right") + ""
  );
  let t4;
  let t5;
  let if_block_anchor;
  let current;
  let mounted;
  let dispose;
  cellmenuicons0 = new CellMenuIcons({ props: { icon: "add-column-left" } });
  cellmenuicons1 = new CellMenuIcons({ props: { icon: "add-column-right" } });
  let if_block = (
    /*can_delete_cols*/
    ctx[7] && create_if_block_1$2(ctx)
  );
  return {
    c() {
      button0 = element("button");
      create_component(cellmenuicons0.$$.fragment);
      t0 = space();
      t1 = text(t1_value);
      t2 = space();
      button1 = element("button");
      create_component(cellmenuicons1.$$.fragment);
      t3 = space();
      t4 = text(t4_value);
      t5 = space();
      if (if_block)
        if_block.c();
      if_block_anchor = empty();
      this.h();
    },
    l(nodes) {
      button0 = claim_element(nodes, "BUTTON", { class: true });
      var button0_nodes = children(button0);
      claim_component(cellmenuicons0.$$.fragment, button0_nodes);
      t0 = claim_space(button0_nodes);
      t1 = claim_text(button0_nodes, t1_value);
      button0_nodes.forEach(detach);
      t2 = claim_space(nodes);
      button1 = claim_element(nodes, "BUTTON", { class: true });
      var button1_nodes = children(button1);
      claim_component(cellmenuicons1.$$.fragment, button1_nodes);
      t3 = claim_space(button1_nodes);
      t4 = claim_text(button1_nodes, t4_value);
      button1_nodes.forEach(detach);
      t5 = claim_space(nodes);
      if (if_block)
        if_block.l(nodes);
      if_block_anchor = empty();
      this.h();
    },
    h() {
      attr(button0, "class", "svelte-9g877");
      attr(button1, "class", "svelte-9g877");
    },
    m(target, anchor) {
      insert_hydration(target, button0, anchor);
      mount_component(cellmenuicons0, button0, null);
      append_hydration(button0, t0);
      append_hydration(button0, t1);
      insert_hydration(target, t2, anchor);
      insert_hydration(target, button1, anchor);
      mount_component(cellmenuicons1, button1, null);
      append_hydration(button1, t3);
      append_hydration(button1, t4);
      insert_hydration(target, t5, anchor);
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            button0,
            "click",
            /*click_handler_2*/
            ctx[20]
          ),
          listen(
            button1,
            "click",
            /*click_handler_3*/
            ctx[21]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if ((!current || dirty & /*i18n*/
      256) && t1_value !== (t1_value = /*i18n*/
      ctx2[8]("dataframe.add_column_left") + ""))
        set_data(t1, t1_value);
      if ((!current || dirty & /*i18n*/
      256) && t4_value !== (t4_value = /*i18n*/
      ctx2[8]("dataframe.add_column_right") + ""))
        set_data(t4, t4_value);
      if (
        /*can_delete_cols*/
        ctx2[7]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*can_delete_cols*/
          128) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block_1$2(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenuicons0.$$.fragment, local);
      transition_in(cellmenuicons1.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(cellmenuicons0.$$.fragment, local);
      transition_out(cellmenuicons1.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button0);
        detach(t2);
        detach(button1);
        detach(t5);
        detach(if_block_anchor);
      }
      destroy_component(cellmenuicons0);
      destroy_component(cellmenuicons1);
      if (if_block)
        if_block.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_if_block_1$2(ctx) {
  let button;
  let cellmenuicons;
  let t0;
  let t1_value = (
    /*i18n*/
    ctx[8]("dataframe.delete_column") + ""
  );
  let t1;
  let current;
  let mounted;
  let dispose;
  cellmenuicons = new CellMenuIcons({ props: { icon: "delete-column" } });
  return {
    c() {
      button = element("button");
      create_component(cellmenuicons.$$.fragment);
      t0 = space();
      t1 = text(t1_value);
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      claim_component(cellmenuicons.$$.fragment, button_nodes);
      t0 = claim_space(button_nodes);
      t1 = claim_text(button_nodes, t1_value);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "delete svelte-9g877");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      mount_component(cellmenuicons, button, null);
      append_hydration(button, t0);
      append_hydration(button, t1);
      current = true;
      if (!mounted) {
        dispose = listen(button, "click", function() {
          if (is_function(
            /*on_delete_col*/
            ctx[5]
          ))
            ctx[5].apply(this, arguments);
        });
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if ((!current || dirty & /*i18n*/
      256) && t1_value !== (t1_value = /*i18n*/
      ctx[8]("dataframe.delete_column") + ""))
        set_data(t1, t1_value);
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenuicons.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(cellmenuicons.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      destroy_component(cellmenuicons);
      mounted = false;
      dispose();
    }
  };
}
function create_fragment$3(ctx) {
  let div;
  let t;
  let current;
  let if_block0 = !/*is_header*/
  ctx[12] && /*can_add_rows*/
  ctx[11] && create_if_block_2$2(ctx);
  let if_block1 = (
    /*can_add_columns*/
    ctx[10] && create_if_block$2(ctx)
  );
  return {
    c() {
      div = element("div");
      if (if_block0)
        if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      if (if_block0)
        if_block0.l(div_nodes);
      t = claim_space(div_nodes);
      if (if_block1)
        if_block1.l(div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "cell-menu svelte-9g877");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      if (if_block0)
        if_block0.m(div, null);
      append_hydration(div, t);
      if (if_block1)
        if_block1.m(div, null);
      ctx[22](div);
      current = true;
    },
    p(ctx2, [dirty]) {
      if (!/*is_header*/
      ctx2[12] && /*can_add_rows*/
      ctx2[11]) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty & /*is_header, can_add_rows*/
          6144) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_2$2(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(div, t);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*can_add_columns*/
        ctx2[10]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty & /*can_add_columns*/
          1024) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block$2(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div, null);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
      ctx[22](null);
    }
  };
}
function instance$3($$self, $$props, $$invalidate) {
  let is_header;
  let can_add_rows;
  let can_add_columns;
  let { x } = $$props;
  let { y } = $$props;
  let { on_add_row_above } = $$props;
  let { on_add_row_below } = $$props;
  let { on_add_column_left } = $$props;
  let { on_add_column_right } = $$props;
  let { row } = $$props;
  let { col_count } = $$props;
  let { row_count } = $$props;
  let { on_delete_row } = $$props;
  let { on_delete_col } = $$props;
  let { can_delete_rows } = $$props;
  let { can_delete_cols } = $$props;
  let { i18n } = $$props;
  let menu_element;
  onMount(() => {
    position_menu();
  });
  function position_menu() {
    if (!menu_element)
      return;
    const viewport_width = window.innerWidth;
    const viewport_height = window.innerHeight;
    const menu_rect = menu_element.getBoundingClientRect();
    let new_x = x - 30;
    let new_y = y - 20;
    if (new_x + menu_rect.width > viewport_width) {
      new_x = x - menu_rect.width + 10;
    }
    if (new_y + menu_rect.height > viewport_height) {
      new_y = y - menu_rect.height + 10;
    }
    $$invalidate(9, menu_element.style.left = `${new_x}px`, menu_element);
    $$invalidate(9, menu_element.style.top = `${new_y}px`, menu_element);
  }
  const click_handler = () => on_add_row_above();
  const click_handler_1 = () => on_add_row_below();
  const click_handler_2 = () => on_add_column_left();
  const click_handler_3 = () => on_add_column_right();
  function div_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      menu_element = $$value;
      $$invalidate(9, menu_element);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("x" in $$props2)
      $$invalidate(13, x = $$props2.x);
    if ("y" in $$props2)
      $$invalidate(14, y = $$props2.y);
    if ("on_add_row_above" in $$props2)
      $$invalidate(0, on_add_row_above = $$props2.on_add_row_above);
    if ("on_add_row_below" in $$props2)
      $$invalidate(1, on_add_row_below = $$props2.on_add_row_below);
    if ("on_add_column_left" in $$props2)
      $$invalidate(2, on_add_column_left = $$props2.on_add_column_left);
    if ("on_add_column_right" in $$props2)
      $$invalidate(3, on_add_column_right = $$props2.on_add_column_right);
    if ("row" in $$props2)
      $$invalidate(15, row = $$props2.row);
    if ("col_count" in $$props2)
      $$invalidate(16, col_count = $$props2.col_count);
    if ("row_count" in $$props2)
      $$invalidate(17, row_count = $$props2.row_count);
    if ("on_delete_row" in $$props2)
      $$invalidate(4, on_delete_row = $$props2.on_delete_row);
    if ("on_delete_col" in $$props2)
      $$invalidate(5, on_delete_col = $$props2.on_delete_col);
    if ("can_delete_rows" in $$props2)
      $$invalidate(6, can_delete_rows = $$props2.can_delete_rows);
    if ("can_delete_cols" in $$props2)
      $$invalidate(7, can_delete_cols = $$props2.can_delete_cols);
    if ("i18n" in $$props2)
      $$invalidate(8, i18n = $$props2.i18n);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*row*/
    32768) {
      $$invalidate(12, is_header = row === -1);
    }
    if ($$self.$$.dirty & /*row_count*/
    131072) {
      $$invalidate(11, can_add_rows = row_count[1] === "dynamic");
    }
    if ($$self.$$.dirty & /*col_count*/
    65536) {
      $$invalidate(10, can_add_columns = col_count[1] === "dynamic");
    }
  };
  return [
    on_add_row_above,
    on_add_row_below,
    on_add_column_left,
    on_add_column_right,
    on_delete_row,
    on_delete_col,
    can_delete_rows,
    can_delete_cols,
    i18n,
    menu_element,
    can_add_columns,
    can_add_rows,
    is_header,
    x,
    y,
    row,
    col_count,
    row_count,
    click_handler,
    click_handler_1,
    click_handler_2,
    click_handler_3,
    div_binding
  ];
}
class CellMenu extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$3, create_fragment$3, safe_not_equal, {
      x: 13,
      y: 14,
      on_add_row_above: 0,
      on_add_row_below: 1,
      on_add_column_left: 2,
      on_add_column_right: 3,
      row: 15,
      col_count: 16,
      row_count: 17,
      on_delete_row: 4,
      on_delete_col: 5,
      can_delete_rows: 6,
      can_delete_cols: 7,
      i18n: 8
    });
  }
}
function create_if_block_2$1(ctx) {
  let button;
  let current_block_type_index;
  let if_block;
  let button_aria_label_value;
  let button_title_value;
  let current;
  let mounted;
  let dispose;
  const if_block_creators = [create_if_block_3$1, create_else_block_1];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*copied*/
      ctx2[3]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      button = element("button");
      if_block.c();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", {
        class: true,
        "aria-label": true,
        title: true
      });
      var button_nodes = children(button);
      if_block.l(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "toolbar-button svelte-1v8q0in");
      attr(button, "aria-label", button_aria_label_value = /*copied*/
      ctx[3] ? "Copied to clipboard" : "Copy table data");
      attr(button, "title", button_title_value = /*copied*/
      ctx[3] ? "Copied to clipboard" : "Copy table data");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if_blocks[current_block_type_index].m(button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*handle_copy*/
          ctx[4]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2);
      if (current_block_type_index !== previous_block_index) {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        }
        transition_in(if_block, 1);
        if_block.m(button, null);
      }
      if (!current || dirty & /*copied*/
      8 && button_aria_label_value !== (button_aria_label_value = /*copied*/
      ctx2[3] ? "Copied to clipboard" : "Copy table data")) {
        attr(button, "aria-label", button_aria_label_value);
      }
      if (!current || dirty & /*copied*/
      8 && button_title_value !== (button_title_value = /*copied*/
      ctx2[3] ? "Copied to clipboard" : "Copy table data")) {
        attr(button, "title", button_title_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      if_blocks[current_block_type_index].d();
      mounted = false;
      dispose();
    }
  };
}
function create_else_block_1(ctx) {
  let copy;
  let current;
  copy = new Copy({});
  return {
    c() {
      create_component(copy.$$.fragment);
    },
    l(nodes) {
      claim_component(copy.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(copy, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(copy.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(copy.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(copy, detaching);
    }
  };
}
function create_if_block_3$1(ctx) {
  let check;
  let current;
  check = new Check({});
  return {
    c() {
      create_component(check.$$.fragment);
    },
    l(nodes) {
      claim_component(check.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(check, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(check.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(check.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(check, detaching);
    }
  };
}
function create_if_block$1(ctx) {
  let button;
  let current_block_type_index;
  let if_block;
  let button_aria_label_value;
  let button_title_value;
  let current;
  let mounted;
  let dispose;
  const if_block_creators = [create_if_block_1$1, create_else_block];
  const if_blocks = [];
  function select_block_type_1(ctx2, dirty) {
    if (
      /*is_fullscreen*/
      ctx2[2]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type_1(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      button = element("button");
      if_block.c();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", {
        class: true,
        "aria-label": true,
        title: true
      });
      var button_nodes = children(button);
      if_block.l(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "toolbar-button svelte-1v8q0in");
      attr(button, "aria-label", button_aria_label_value = /*is_fullscreen*/
      ctx[2] ? "Exit fullscreen" : "Enter fullscreen");
      attr(button, "title", button_title_value = /*is_fullscreen*/
      ctx[2] ? "Exit fullscreen" : "Enter fullscreen");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if_blocks[current_block_type_index].m(button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*click_handler*/
          ctx[6]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type_1(ctx2);
      if (current_block_type_index !== previous_block_index) {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        }
        transition_in(if_block, 1);
        if_block.m(button, null);
      }
      if (!current || dirty & /*is_fullscreen*/
      4 && button_aria_label_value !== (button_aria_label_value = /*is_fullscreen*/
      ctx2[2] ? "Exit fullscreen" : "Enter fullscreen")) {
        attr(button, "aria-label", button_aria_label_value);
      }
      if (!current || dirty & /*is_fullscreen*/
      4 && button_title_value !== (button_title_value = /*is_fullscreen*/
      ctx2[2] ? "Exit fullscreen" : "Enter fullscreen")) {
        attr(button, "title", button_title_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      if_blocks[current_block_type_index].d();
      mounted = false;
      dispose();
    }
  };
}
function create_else_block(ctx) {
  let maximize;
  let current;
  maximize = new Maximize({});
  return {
    c() {
      create_component(maximize.$$.fragment);
    },
    l(nodes) {
      claim_component(maximize.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(maximize, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(maximize.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(maximize.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(maximize, detaching);
    }
  };
}
function create_if_block_1$1(ctx) {
  let minimize;
  let current;
  minimize = new Minimize({});
  return {
    c() {
      create_component(minimize.$$.fragment);
    },
    l(nodes) {
      claim_component(minimize.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(minimize, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(minimize.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(minimize.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(minimize, detaching);
    }
  };
}
function create_fragment$2(ctx) {
  let div;
  let t;
  let current;
  let if_block0 = (
    /*show_copy_button*/
    ctx[1] && create_if_block_2$1(ctx)
  );
  let if_block1 = (
    /*show_fullscreen_button*/
    ctx[0] && create_if_block$1(ctx)
  );
  return {
    c() {
      div = element("div");
      if (if_block0)
        if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {
        class: true,
        role: true,
        "aria-label": true
      });
      var div_nodes = children(div);
      if (if_block0)
        if_block0.l(div_nodes);
      t = claim_space(div_nodes);
      if (if_block1)
        if_block1.l(div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "toolbar svelte-1v8q0in");
      attr(div, "role", "toolbar");
      attr(div, "aria-label", "Table actions");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      if (if_block0)
        if_block0.m(div, null);
      append_hydration(div, t);
      if (if_block1)
        if_block1.m(div, null);
      current = true;
    },
    p(ctx2, [dirty]) {
      if (
        /*show_copy_button*/
        ctx2[1]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty & /*show_copy_button*/
          2) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_2$1(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(div, t);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*show_fullscreen_button*/
        ctx2[0]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty & /*show_fullscreen_button*/
          1) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block$1(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div, null);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
    }
  };
}
function instance$2($$self, $$props, $$invalidate) {
  let { show_fullscreen_button = false } = $$props;
  let { show_copy_button = false } = $$props;
  let { is_fullscreen = false } = $$props;
  let { on_copy } = $$props;
  let copied = false;
  let timer;
  function copy_feedback() {
    $$invalidate(3, copied = true);
    if (timer)
      clearTimeout(timer);
    timer = setTimeout(
      () => {
        $$invalidate(3, copied = false);
      },
      2e3
    );
  }
  async function handle_copy() {
    await on_copy();
    copy_feedback();
  }
  onDestroy(() => {
    if (timer)
      clearTimeout(timer);
  });
  function click_handler(event) {
    bubble.call(this, $$self, event);
  }
  $$self.$$set = ($$props2) => {
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(0, show_fullscreen_button = $$props2.show_fullscreen_button);
    if ("show_copy_button" in $$props2)
      $$invalidate(1, show_copy_button = $$props2.show_copy_button);
    if ("is_fullscreen" in $$props2)
      $$invalidate(2, is_fullscreen = $$props2.is_fullscreen);
    if ("on_copy" in $$props2)
      $$invalidate(5, on_copy = $$props2.on_copy);
  };
  return [
    show_fullscreen_button,
    show_copy_button,
    is_fullscreen,
    copied,
    handle_copy,
    on_copy,
    click_handler
  ];
}
class Toolbar extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$2, create_fragment$2, safe_not_equal, {
      show_fullscreen_button: 0,
      show_copy_button: 1,
      is_fullscreen: 2,
      on_copy: 5
    });
  }
}
async function copy_table_data(data, headers) {
  const header_row = headers ? headers.map((h) => String(h.value)).join(",") : "";
  const table_data = data.map((row) => row.map((cell) => String(cell.value)).join(",")).join("\n");
  const all_data = header_row ? `${header_row}
${table_data}` : table_data;
  try {
    if ("clipboard" in navigator) {
      await navigator.clipboard.writeText(all_data);
    } else {
      const textArea = document.createElement("textarea");
      textArea.value = all_data;
      textArea.style.position = "absolute";
      textArea.style.left = "-999999px";
      document.body.prepend(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }
  } catch (error) {
    console.error("Failed to copy table data:", error);
  }
}
const { window: window_1 } = globals;
function get_each_context_1(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[124] = list[i].value;
  child_ctx[125] = list[i].id;
  child_ctx[128] = list;
  child_ctx[129] = i;
  return child_ctx;
}
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[124] = list[i].value;
  child_ctx[125] = list[i].id;
  child_ctx[126] = list;
  child_ctx[127] = i;
  return child_ctx;
}
function get_each_context_2(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[124] = list[i].value;
  child_ctx[125] = list[i].id;
  child_ctx[130] = list;
  child_ctx[127] = i;
  return child_ctx;
}
function get_each_context_3(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[124] = list[i].value;
  child_ctx[125] = list[i].id;
  child_ctx[129] = i;
  return child_ctx;
}
function create_if_block_9(ctx) {
  let div;
  let p;
  let t;
  return {
    c() {
      div = element("div");
      p = element("p");
      t = text(
        /*label*/
        ctx[1]
      );
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      p = claim_element(div_nodes, "P", { class: true });
      var p_nodes = children(p);
      t = claim_text(
        p_nodes,
        /*label*/
        ctx[1]
      );
      p_nodes.forEach(detach);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(p, "class", "svelte-1l6bino");
      attr(div, "class", "label svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      append_hydration(div, p);
      append_hydration(p, t);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      2)
        set_data(
          t,
          /*label*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
    }
  };
}
function create_if_block_8(ctx) {
  let caption;
  let t;
  return {
    c() {
      caption = element("caption");
      t = text(
        /*label*/
        ctx[1]
      );
      this.h();
    },
    l(nodes) {
      caption = claim_element(nodes, "CAPTION", { class: true });
      var caption_nodes = children(caption);
      t = claim_text(
        caption_nodes,
        /*label*/
        ctx[1]
      );
      caption_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(caption, "class", "sr-only");
    },
    m(target, anchor) {
      insert_hydration(target, caption, anchor);
      append_hydration(caption, t);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      2)
        set_data(
          t,
          /*label*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(caption);
      }
    }
  };
}
function create_if_block_7(ctx) {
  let th;
  return {
    c() {
      th = element("th");
      this.h();
    },
    l(nodes) {
      th = claim_element(nodes, "TH", { class: true });
      children(th).forEach(detach);
      this.h();
    },
    h() {
      attr(th, "class", "row-number-header svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, th, anchor);
    },
    d(detaching) {
      if (detaching) {
        detach(th);
      }
    }
  };
}
function create_each_block_3(key_1, ctx) {
  let th;
  let div1;
  let editablecell;
  let t0;
  let div0;
  let svg;
  let path;
  let div0_class_value;
  let t1;
  let th_aria_sort_value;
  let current;
  editablecell = new EditableCell({
    props: {
      value: (
        /*value*/
        ctx[124]
      ),
      latex_delimiters: (
        /*latex_delimiters*/
        ctx[5]
      ),
      line_breaks: (
        /*line_breaks*/
        ctx[11]
      ),
      header: true,
      edit: false,
      el: null,
      root: (
        /*root*/
        ctx[8]
      )
    }
  });
  return {
    key: key_1,
    first: null,
    c() {
      th = element("th");
      div1 = element("div");
      create_component(editablecell.$$.fragment);
      t0 = space();
      div0 = element("div");
      svg = svg_element("svg");
      path = svg_element("path");
      t1 = space();
      this.h();
    },
    l(nodes) {
      th = claim_element(nodes, "TH", { "aria-sort": true, class: true });
      var th_nodes = children(th);
      div1 = claim_element(th_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      claim_component(editablecell.$$.fragment, div1_nodes);
      t0 = claim_space(div1_nodes);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      svg = claim_svg_element(div0_nodes, "svg", {
        width: true,
        height: true,
        viewBox: true,
        fill: true,
        xmlns: true,
        class: true
      });
      var svg_nodes = children(svg);
      path = claim_svg_element(svg_nodes, "path", { d: true });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      div0_nodes.forEach(detach);
      div1_nodes.forEach(detach);
      t1 = claim_space(th_nodes);
      th_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(path, "d", "M4.49999 0L8.3971 6.75H0.602875L4.49999 0Z");
      attr(svg, "width", "1em");
      attr(svg, "height", "1em");
      attr(svg, "viewBox", "0 0 9 7");
      attr(svg, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "class", "svelte-1l6bino");
      attr(div0, "class", div0_class_value = "sort-button " + /*sort_direction*/
      ctx[23] + " svelte-1l6bino");
      toggle_class(
        div0,
        "sorted",
        /*sort_by*/
        ctx[24] === /*i*/
        ctx[129]
      );
      toggle_class(
        div0,
        "des",
        /*sort_by*/
        ctx[24] === /*i*/
        ctx[129] && /*sort_direction*/
        ctx[23] === "des"
      );
      attr(div1, "class", "cell-wrap svelte-1l6bino");
      attr(th, "aria-sort", th_aria_sort_value = /*get_sort_status*/
      ctx[43](
        /*value*/
        ctx[124],
        /*sort_by*/
        ctx[24],
        /*sort_direction*/
        ctx[23]
      ));
      attr(th, "class", "svelte-1l6bino");
      toggle_class(
        th,
        "editing",
        /*header_edit*/
        ctx[31] === /*i*/
        ctx[129]
      );
      set_style(
        th,
        "width",
        /*column_widths*/
        ctx[12].length ? (
          /*column_widths*/
          ctx[12][
            /*i*/
            ctx[129]
          ]
        ) : void 0
      );
      this.first = th;
    },
    m(target, anchor) {
      insert_hydration(target, th, anchor);
      append_hydration(th, div1);
      mount_component(editablecell, div1, null);
      append_hydration(div1, t0);
      append_hydration(div1, div0);
      append_hydration(div0, svg);
      append_hydration(svg, path);
      append_hydration(th, t1);
      current = true;
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      const editablecell_changes = {};
      if (dirty[0] & /*_headers*/
      2097152)
        editablecell_changes.value = /*value*/
        ctx[124];
      if (dirty[0] & /*latex_delimiters*/
      32)
        editablecell_changes.latex_delimiters = /*latex_delimiters*/
        ctx[5];
      if (dirty[0] & /*line_breaks*/
      2048)
        editablecell_changes.line_breaks = /*line_breaks*/
        ctx[11];
      if (dirty[0] & /*root*/
      256)
        editablecell_changes.root = /*root*/
        ctx[8];
      editablecell.$set(editablecell_changes);
      if (!current || dirty[0] & /*sort_direction*/
      8388608 && div0_class_value !== (div0_class_value = "sort-button " + /*sort_direction*/
      ctx[23] + " svelte-1l6bino")) {
        attr(div0, "class", div0_class_value);
      }
      if (!current || dirty[0] & /*sort_direction, sort_by, _headers*/
      27262976) {
        toggle_class(
          div0,
          "sorted",
          /*sort_by*/
          ctx[24] === /*i*/
          ctx[129]
        );
      }
      if (!current || dirty[0] & /*sort_direction, sort_by, _headers, sort_direction*/
      27262976) {
        toggle_class(
          div0,
          "des",
          /*sort_by*/
          ctx[24] === /*i*/
          ctx[129] && /*sort_direction*/
          ctx[23] === "des"
        );
      }
      if (!current || dirty[0] & /*_headers, sort_by, sort_direction*/
      27262976 && th_aria_sort_value !== (th_aria_sort_value = /*get_sort_status*/
      ctx[43](
        /*value*/
        ctx[124],
        /*sort_by*/
        ctx[24],
        /*sort_direction*/
        ctx[23]
      ))) {
        attr(th, "aria-sort", th_aria_sort_value);
      }
      if (!current || dirty[0] & /*_headers*/
      2097152 | dirty[1] & /*header_edit*/
      1) {
        toggle_class(
          th,
          "editing",
          /*header_edit*/
          ctx[31] === /*i*/
          ctx[129]
        );
      }
      if (dirty[0] & /*column_widths, _headers*/
      2101248) {
        set_style(
          th,
          "width",
          /*column_widths*/
          ctx[12].length ? (
            /*column_widths*/
            ctx[12][
              /*i*/
              ctx[129]
            ]
          ) : void 0
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(editablecell.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(editablecell.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(th);
      }
      destroy_component(editablecell);
    }
  };
}
function create_each_block_2(key_1, ctx) {
  let td;
  let div;
  let editablecell;
  let t;
  let j = (
    /*j*/
    ctx[127]
  );
  let current;
  editablecell = new EditableCell({
    props: {
      value: (
        /*value*/
        ctx[124]
      ),
      latex_delimiters: (
        /*latex_delimiters*/
        ctx[5]
      ),
      line_breaks: (
        /*line_breaks*/
        ctx[11]
      ),
      datatype: Array.isArray(
        /*datatype*/
        ctx[0]
      ) ? (
        /*datatype*/
        ctx[0][
          /*j*/
          ctx[127]
        ]
      ) : (
        /*datatype*/
        ctx[0]
      ),
      edit: false,
      el: null,
      root: (
        /*root*/
        ctx[8]
      )
    }
  });
  const assign_td = () => (
    /*td_binding*/
    ctx[67](td, j)
  );
  const unassign_td = () => (
    /*td_binding*/
    ctx[67](null, j)
  );
  return {
    key: key_1,
    first: null,
    c() {
      td = element("td");
      div = element("div");
      create_component(editablecell.$$.fragment);
      t = space();
      this.h();
    },
    l(nodes) {
      td = claim_element(nodes, "TD", { tabindex: true, class: true });
      var td_nodes = children(td);
      div = claim_element(td_nodes, "DIV", { class: true });
      var div_nodes = children(div);
      claim_component(editablecell.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      t = claim_space(td_nodes);
      td_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "cell-wrap svelte-1l6bino");
      attr(td, "tabindex", "-1");
      attr(td, "class", "svelte-1l6bino");
      this.first = td;
    },
    m(target, anchor) {
      insert_hydration(target, td, anchor);
      append_hydration(td, div);
      mount_component(editablecell, div, null);
      append_hydration(td, t);
      assign_td();
      current = true;
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      const editablecell_changes = {};
      if (dirty[1] & /*max*/
      2048)
        editablecell_changes.value = /*value*/
        ctx[124];
      if (dirty[0] & /*latex_delimiters*/
      32)
        editablecell_changes.latex_delimiters = /*latex_delimiters*/
        ctx[5];
      if (dirty[0] & /*line_breaks*/
      2048)
        editablecell_changes.line_breaks = /*line_breaks*/
        ctx[11];
      if (dirty[0] & /*datatype*/
      1 | dirty[1] & /*max*/
      2048)
        editablecell_changes.datatype = Array.isArray(
          /*datatype*/
          ctx[0]
        ) ? (
          /*datatype*/
          ctx[0][
            /*j*/
            ctx[127]
          ]
        ) : (
          /*datatype*/
          ctx[0]
        );
      if (dirty[0] & /*root*/
      256)
        editablecell_changes.root = /*root*/
        ctx[8];
      editablecell.$set(editablecell_changes);
      if (j !== /*j*/
      ctx[127]) {
        unassign_td();
        j = /*j*/
        ctx[127];
        assign_td();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(editablecell.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(editablecell.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(td);
      }
      destroy_component(editablecell);
      unassign_td();
    }
  };
}
function create_if_block_6(ctx) {
  let caption;
  let t;
  return {
    c() {
      caption = element("caption");
      t = text(
        /*label*/
        ctx[1]
      );
      this.h();
    },
    l(nodes) {
      caption = claim_element(nodes, "CAPTION", { class: true });
      var caption_nodes = children(caption);
      t = claim_text(
        caption_nodes,
        /*label*/
        ctx[1]
      );
      caption_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(caption, "class", "sr-only");
    },
    m(target, anchor) {
      insert_hydration(target, caption, anchor);
      append_hydration(caption, t);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      2)
        set_data(
          t,
          /*label*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(caption);
      }
    }
  };
}
function create_default_slot_1(ctx) {
  let if_block_anchor;
  let if_block = (
    /*label*/
    ctx[1] && /*label*/
    ctx[1].length !== 0 && create_if_block_6(ctx)
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
    },
    p(ctx2, dirty) {
      if (
        /*label*/
        ctx2[1] && /*label*/
        ctx2[1].length !== 0
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block_6(ctx2);
          if_block.c();
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
    },
    d(detaching) {
      if (detaching) {
        detach(if_block_anchor);
      }
      if (if_block)
        if_block.d(detaching);
    }
  };
}
function create_if_block_5(ctx) {
  let th;
  return {
    c() {
      th = element("th");
      this.h();
    },
    l(nodes) {
      th = claim_element(nodes, "TH", { class: true });
      children(th).forEach(detach);
      this.h();
    },
    h() {
      attr(th, "class", "row-number-header svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, th, anchor);
    },
    d(detaching) {
      if (detaching) {
        detach(th);
      }
    }
  };
}
function create_if_block_4(ctx) {
  let button;
  let textContent = "⋮";
  let mounted;
  let dispose;
  function click_handler_3(...args) {
    return (
      /*click_handler_3*/
      ctx[80](
        /*i*/
        ctx[129],
        ...args
      )
    );
  }
  return {
    c() {
      button = element("button");
      button.textContent = textContent;
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button) !== "svelte-1e92k9i")
        button.textContent = textContent;
      this.h();
    },
    h() {
      attr(button, "class", "cell-menu-button svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if (!mounted) {
        dispose = listen(button, "click", click_handler_3);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      mounted = false;
      dispose();
    }
  };
}
function create_each_block_1(key_1, ctx) {
  let th;
  let div1;
  let div0;
  let editablecell;
  let updating_value;
  let updating_el;
  let t0;
  let button;
  let svg;
  let path;
  let button_class_value;
  let t1;
  let t2;
  let th_aria_sort_value;
  let current;
  let mounted;
  let dispose;
  function editablecell_value_binding_1(value) {
    ctx[76](
      value,
      /*i*/
      ctx[129]
    );
  }
  function editablecell_el_binding_1(value) {
    ctx[77](
      value,
      /*id*/
      ctx[125]
    );
  }
  function dblclick_handler() {
    return (
      /*dblclick_handler*/
      ctx[78](
        /*i*/
        ctx[129]
      )
    );
  }
  let editablecell_props = {
    latex_delimiters: (
      /*latex_delimiters*/
      ctx[5]
    ),
    line_breaks: (
      /*line_breaks*/
      ctx[11]
    ),
    edit: (
      /*header_edit*/
      ctx[31] === /*i*/
      ctx[129]
    ),
    header: true,
    root: (
      /*root*/
      ctx[8]
    )
  };
  if (
    /*_headers*/
    ctx[21][
      /*i*/
      ctx[129]
    ].value !== void 0
  ) {
    editablecell_props.value = /*_headers*/
    ctx[21][
      /*i*/
      ctx[129]
    ].value;
  }
  if (
    /*els*/
    ctx[29][
      /*id*/
      ctx[125]
    ].input !== void 0
  ) {
    editablecell_props.el = /*els*/
    ctx[29][
      /*id*/
      ctx[125]
    ].input;
  }
  editablecell = new EditableCell({ props: editablecell_props });
  binding_callbacks.push(() => bind(editablecell, "value", editablecell_value_binding_1));
  binding_callbacks.push(() => bind(editablecell, "el", editablecell_el_binding_1));
  editablecell.$on(
    "keydown",
    /*end_header_edit*/
    ctx[47]
  );
  editablecell.$on("dblclick", dblclick_handler);
  function click_handler_2(...args) {
    return (
      /*click_handler_2*/
      ctx[79](
        /*i*/
        ctx[129],
        ...args
      )
    );
  }
  let if_block = (
    /*editable*/
    ctx[6] && create_if_block_4(ctx)
  );
  function click_handler_4() {
    return (
      /*click_handler_4*/
      ctx[81](
        /*i*/
        ctx[129]
      )
    );
  }
  return {
    key: key_1,
    first: null,
    c() {
      th = element("th");
      div1 = element("div");
      div0 = element("div");
      create_component(editablecell.$$.fragment);
      t0 = space();
      button = element("button");
      svg = svg_element("svg");
      path = svg_element("path");
      t1 = space();
      if (if_block)
        if_block.c();
      t2 = space();
      this.h();
    },
    l(nodes) {
      th = claim_element(nodes, "TH", {
        "aria-sort": true,
        style: true,
        class: true
      });
      var th_nodes = children(th);
      div1 = claim_element(th_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      claim_component(editablecell.$$.fragment, div0_nodes);
      t0 = claim_space(div0_nodes);
      button = claim_element(div0_nodes, "BUTTON", { class: true, tabindex: true });
      var button_nodes = children(button);
      svg = claim_svg_element(button_nodes, "svg", {
        width: true,
        height: true,
        viewBox: true,
        fill: true,
        xmlns: true,
        class: true
      });
      var svg_nodes = children(svg);
      path = claim_svg_element(svg_nodes, "path", { d: true });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      button_nodes.forEach(detach);
      div0_nodes.forEach(detach);
      t1 = claim_space(div1_nodes);
      if (if_block)
        if_block.l(div1_nodes);
      div1_nodes.forEach(detach);
      t2 = claim_space(th_nodes);
      th_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(path, "d", "M4.49999 0L8.3971 6.75H0.602875L4.49999 0Z");
      attr(svg, "width", "1em");
      attr(svg, "height", "1em");
      attr(svg, "viewBox", "0 0 9 7");
      attr(svg, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "class", "svelte-1l6bino");
      attr(button, "class", button_class_value = "sort-button " + /*sort_direction*/
      ctx[23] + " svelte-1l6bino");
      attr(button, "tabindex", "0");
      toggle_class(
        button,
        "sorted",
        /*sort_by*/
        ctx[24] === /*i*/
        ctx[129]
      );
      toggle_class(
        button,
        "des",
        /*sort_by*/
        ctx[24] === /*i*/
        ctx[129] && /*sort_direction*/
        ctx[23] === "des"
      );
      attr(div0, "class", "header-content svelte-1l6bino");
      attr(div1, "class", "cell-wrap svelte-1l6bino");
      attr(th, "aria-sort", th_aria_sort_value = /*get_sort_status*/
      ctx[43](
        /*value*/
        ctx[124],
        /*sort_by*/
        ctx[24],
        /*sort_direction*/
        ctx[23]
      ));
      set_style(th, "width", "var(--cell-width-" + /*i*/
      ctx[129] + ")");
      attr(th, "class", "svelte-1l6bino");
      toggle_class(
        th,
        "focus",
        /*header_edit*/
        ctx[31] === /*i*/
        ctx[129] || /*selected_header*/
        ctx[32] === /*i*/
        ctx[129]
      );
      this.first = th;
    },
    m(target, anchor) {
      insert_hydration(target, th, anchor);
      append_hydration(th, div1);
      append_hydration(div1, div0);
      mount_component(editablecell, div0, null);
      append_hydration(div0, t0);
      append_hydration(div0, button);
      append_hydration(button, svg);
      append_hydration(svg, path);
      append_hydration(div1, t1);
      if (if_block)
        if_block.m(div1, null);
      append_hydration(th, t2);
      current = true;
      if (!mounted) {
        dispose = [
          listen(button, "click", click_handler_2),
          listen(th, "click", click_handler_4)
        ];
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      const editablecell_changes = {};
      if (dirty[0] & /*latex_delimiters*/
      32)
        editablecell_changes.latex_delimiters = /*latex_delimiters*/
        ctx[5];
      if (dirty[0] & /*line_breaks*/
      2048)
        editablecell_changes.line_breaks = /*line_breaks*/
        ctx[11];
      if (dirty[0] & /*_headers*/
      2097152 | dirty[1] & /*header_edit*/
      1)
        editablecell_changes.edit = /*header_edit*/
        ctx[31] === /*i*/
        ctx[129];
      if (dirty[0] & /*root*/
      256)
        editablecell_changes.root = /*root*/
        ctx[8];
      if (!updating_value && dirty[0] & /*_headers*/
      2097152) {
        updating_value = true;
        editablecell_changes.value = /*_headers*/
        ctx[21][
          /*i*/
          ctx[129]
        ].value;
        add_flush_callback(() => updating_value = false);
      }
      if (!updating_el && dirty[0] & /*els, _headers*/
      538968064) {
        updating_el = true;
        editablecell_changes.el = /*els*/
        ctx[29][
          /*id*/
          ctx[125]
        ].input;
        add_flush_callback(() => updating_el = false);
      }
      editablecell.$set(editablecell_changes);
      if (!current || dirty[0] & /*sort_direction*/
      8388608 && button_class_value !== (button_class_value = "sort-button " + /*sort_direction*/
      ctx[23] + " svelte-1l6bino")) {
        attr(button, "class", button_class_value);
      }
      if (!current || dirty[0] & /*sort_direction, sort_by, _headers*/
      27262976) {
        toggle_class(
          button,
          "sorted",
          /*sort_by*/
          ctx[24] === /*i*/
          ctx[129]
        );
      }
      if (!current || dirty[0] & /*sort_direction, sort_by, _headers, sort_direction*/
      27262976) {
        toggle_class(
          button,
          "des",
          /*sort_by*/
          ctx[24] === /*i*/
          ctx[129] && /*sort_direction*/
          ctx[23] === "des"
        );
      }
      if (
        /*editable*/
        ctx[6]
      ) {
        if (if_block) {
          if_block.p(ctx, dirty);
        } else {
          if_block = create_if_block_4(ctx);
          if_block.c();
          if_block.m(div1, null);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (!current || dirty[0] & /*_headers, sort_by, sort_direction*/
      27262976 && th_aria_sort_value !== (th_aria_sort_value = /*get_sort_status*/
      ctx[43](
        /*value*/
        ctx[124],
        /*sort_by*/
        ctx[24],
        /*sort_direction*/
        ctx[23]
      ))) {
        attr(th, "aria-sort", th_aria_sort_value);
      }
      if (!current || dirty[0] & /*_headers*/
      2097152) {
        set_style(th, "width", "var(--cell-width-" + /*i*/
        ctx[129] + ")");
      }
      if (!current || dirty[0] & /*_headers*/
      2097152 | dirty[1] & /*header_edit, selected_header*/
      3) {
        toggle_class(
          th,
          "focus",
          /*header_edit*/
          ctx[31] === /*i*/
          ctx[129] || /*selected_header*/
          ctx[32] === /*i*/
          ctx[129]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(editablecell.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(editablecell.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(th);
      }
      destroy_component(editablecell);
      if (if_block)
        if_block.d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_thead_slot(ctx) {
  let tr;
  let t;
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let current;
  let if_block = (
    /*show_row_numbers*/
    ctx[13] && create_if_block_5()
  );
  let each_value_1 = ensure_array_like(
    /*_headers*/
    ctx[21]
  );
  const get_key = (ctx2) => (
    /*id*/
    ctx2[125]
  );
  for (let i = 0; i < each_value_1.length; i += 1) {
    let child_ctx = get_each_context_1(ctx, each_value_1, i);
    let key = get_key(child_ctx);
    each_1_lookup.set(key, each_blocks[i] = create_each_block_1(key, child_ctx));
  }
  return {
    c() {
      tr = element("tr");
      if (if_block)
        if_block.c();
      t = space();
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      this.h();
    },
    l(nodes) {
      tr = claim_element(nodes, "TR", { slot: true, class: true });
      var tr_nodes = children(tr);
      if (if_block)
        if_block.l(tr_nodes);
      t = claim_space(tr_nodes);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(tr_nodes);
      }
      tr_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(tr, "slot", "thead");
      attr(tr, "class", "svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, tr, anchor);
      if (if_block)
        if_block.m(tr, null);
      append_hydration(tr, t);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(tr, null);
        }
      }
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*show_row_numbers*/
        ctx2[13]
      ) {
        if (if_block)
          ;
        else {
          if_block = create_if_block_5();
          if_block.c();
          if_block.m(tr, t);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (dirty[0] & /*_headers, sort_by, sort_direction, editable, latex_delimiters, line_breaks, root, els*/
      564136288 | dirty[1] & /*get_sort_status, header_edit, selected_header, toggle_header_button, toggle_header_menu, handle_sort, end_header_edit, edit_header*/
      71421955) {
        each_value_1 = ensure_array_like(
          /*_headers*/
          ctx2[21]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value_1, each_1_lookup, tr, outro_and_destroy_block, create_each_block_1, null, get_each_context_1);
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      for (let i = 0; i < each_value_1.length; i += 1) {
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
        detach(tr);
      }
      if (if_block)
        if_block.d();
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d();
      }
    }
  };
}
function create_if_block_3(ctx) {
  let td;
  let t_value = (
    /*index*/
    ctx[122] + 1 + ""
  );
  let t;
  let td_title_value;
  return {
    c() {
      td = element("td");
      t = text(t_value);
      this.h();
    },
    l(nodes) {
      td = claim_element(nodes, "TD", { class: true, title: true });
      var td_nodes = children(td);
      t = claim_text(td_nodes, t_value);
      td_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(td, "class", "row-number svelte-1l6bino");
      attr(td, "title", td_title_value = `Row ${/*index*/
      ctx[122] + 1}`);
    },
    m(target, anchor) {
      insert_hydration(target, td, anchor);
      append_hydration(td, t);
    },
    p(ctx2, dirty) {
      if (dirty[3] & /*index*/
      536870912 && t_value !== (t_value = /*index*/
      ctx2[122] + 1 + ""))
        set_data(t, t_value);
      if (dirty[3] & /*index*/
      536870912 && td_title_value !== (td_title_value = `Row ${/*index*/
      ctx2[122] + 1}`)) {
        attr(td, "title", td_title_value);
      }
    },
    d(detaching) {
      if (detaching) {
        detach(td);
      }
    }
  };
}
function create_if_block_2(ctx) {
  let button;
  let textContent = "⋮";
  let mounted;
  let dispose;
  function click_handler(...args) {
    return (
      /*click_handler*/
      ctx[73](
        /*index*/
        ctx[122],
        /*j*/
        ctx[127],
        ...args
      )
    );
  }
  return {
    c() {
      button = element("button");
      button.textContent = textContent;
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button) !== "svelte-b4o3eq")
        button.textContent = textContent;
      this.h();
    },
    h() {
      attr(button, "class", "cell-menu-button svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if (!mounted) {
        dispose = listen(button, "click", click_handler);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      mounted = false;
      dispose();
    }
  };
}
function create_each_block(key_1, ctx) {
  var _a, _b;
  let td;
  let div;
  let editablecell;
  let updating_value;
  let updating_el;
  let t0;
  let t1;
  let td_style_value;
  let style_width = `var(--cell-width-${/*j*/
  ctx[127]})`;
  let current;
  let mounted;
  let dispose;
  function editablecell_value_binding(value) {
    ctx[70](
      value,
      /*index*/
      ctx[122],
      /*j*/
      ctx[127]
    );
  }
  function editablecell_el_binding(value) {
    ctx[71](
      value,
      /*id*/
      ctx[125]
    );
  }
  let editablecell_props = {
    display_value: (
      /*display_value*/
      (_b = (_a = ctx[18]) == null ? void 0 : _a[
        /*index*/
        ctx[122]
      ]) == null ? void 0 : _b[
        /*j*/
        ctx[127]
      ]
    ),
    latex_delimiters: (
      /*latex_delimiters*/
      ctx[5]
    ),
    line_breaks: (
      /*line_breaks*/
      ctx[11]
    ),
    editable: (
      /*editable*/
      ctx[6]
    ),
    edit: dequal(
      /*editing*/
      ctx[28],
      [
        /*index*/
        ctx[122],
        /*j*/
        ctx[127]
      ]
    ),
    datatype: Array.isArray(
      /*datatype*/
      ctx[0]
    ) ? (
      /*datatype*/
      ctx[0][
        /*j*/
        ctx[127]
      ]
    ) : (
      /*datatype*/
      ctx[0]
    ),
    clear_on_focus: (
      /*clear_on_focus*/
      ctx[30]
    ),
    root: (
      /*root*/
      ctx[8]
    )
  };
  if (
    /*data*/
    ctx[22][
      /*index*/
      ctx[122]
    ][
      /*j*/
      ctx[127]
    ].value !== void 0
  ) {
    editablecell_props.value = /*data*/
    ctx[22][
      /*index*/
      ctx[122]
    ][
      /*j*/
      ctx[127]
    ].value;
  }
  if (
    /*els*/
    ctx[29][
      /*id*/
      ctx[125]
    ].input !== void 0
  ) {
    editablecell_props.el = /*els*/
    ctx[29][
      /*id*/
      ctx[125]
    ].input;
  }
  editablecell = new EditableCell({ props: editablecell_props });
  binding_callbacks.push(() => bind(editablecell, "value", editablecell_value_binding));
  binding_callbacks.push(() => bind(editablecell, "el", editablecell_el_binding));
  editablecell.$on(
    "blur",
    /*blur_handler*/
    ctx[72]
  );
  let if_block = (
    /*editable*/
    ctx[6] && create_if_block_2(ctx)
  );
  function touchstart_handler(...args) {
    return (
      /*touchstart_handler*/
      ctx[74](
        /*index*/
        ctx[122],
        /*j*/
        ctx[127],
        ...args
      )
    );
  }
  function click_handler_1(...args) {
    return (
      /*click_handler_1*/
      ctx[75](
        /*index*/
        ctx[122],
        /*j*/
        ctx[127],
        ...args
      )
    );
  }
  return {
    key: key_1,
    first: null,
    c() {
      td = element("td");
      div = element("div");
      create_component(editablecell.$$.fragment);
      t0 = space();
      if (if_block)
        if_block.c();
      t1 = space();
      this.h();
    },
    l(nodes) {
      td = claim_element(nodes, "TD", { tabindex: true, style: true, class: true });
      var td_nodes = children(td);
      div = claim_element(td_nodes, "DIV", { class: true });
      var div_nodes = children(div);
      claim_component(editablecell.$$.fragment, div_nodes);
      t0 = claim_space(div_nodes);
      if (if_block)
        if_block.l(div_nodes);
      div_nodes.forEach(detach);
      t1 = claim_space(td_nodes);
      td_nodes.forEach(detach);
      this.h();
    },
    h() {
      var _a2, _b2;
      attr(div, "class", "cell-wrap svelte-1l6bino");
      attr(td, "tabindex", "0");
      attr(td, "style", td_style_value = /*styling*/
      ((_b2 = (_a2 = ctx[19]) == null ? void 0 : _a2[
        /*index*/
        ctx[122]
      ]) == null ? void 0 : _b2[
        /*j*/
        ctx[127]
      ]) || "");
      attr(td, "class", "svelte-1l6bino");
      toggle_class(td, "focus", dequal(
        /*selected*/
        ctx[20],
        [
          /*index*/
          ctx[122],
          /*j*/
          ctx[127]
        ]
      ));
      toggle_class(
        td,
        "menu-active",
        /*active_cell_menu*/
        ctx[38] && /*active_cell_menu*/
        ctx[38].row === /*index*/
        ctx[122] && /*active_cell_menu*/
        ctx[38].col === /*j*/
        ctx[127]
      );
      set_style(td, "width", style_width);
      this.first = td;
    },
    m(target, anchor) {
      insert_hydration(target, td, anchor);
      append_hydration(td, div);
      mount_component(editablecell, div, null);
      append_hydration(div, t0);
      if (if_block)
        if_block.m(div, null);
      append_hydration(td, t1);
      current = true;
      if (!mounted) {
        dispose = [
          listen(td, "touchstart", touchstart_handler),
          listen(td, "mousedown", mousedown_handler),
          listen(td, "click", click_handler_1)
        ];
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      var _a2, _b2, _c, _d;
      ctx = new_ctx;
      const editablecell_changes = {};
      if (dirty[0] & /*display_value*/
      262144 | dirty[3] & /*index, item*/
      1610612736)
        editablecell_changes.display_value = /*display_value*/
        (_b2 = (_a2 = ctx[18]) == null ? void 0 : _a2[
          /*index*/
          ctx[122]
        ]) == null ? void 0 : _b2[
          /*j*/
          ctx[127]
        ];
      if (dirty[0] & /*latex_delimiters*/
      32)
        editablecell_changes.latex_delimiters = /*latex_delimiters*/
        ctx[5];
      if (dirty[0] & /*line_breaks*/
      2048)
        editablecell_changes.line_breaks = /*line_breaks*/
        ctx[11];
      if (dirty[0] & /*editable*/
      64)
        editablecell_changes.editable = /*editable*/
        ctx[6];
      if (dirty[0] & /*editing*/
      268435456 | dirty[3] & /*index, item*/
      1610612736)
        editablecell_changes.edit = dequal(
          /*editing*/
          ctx[28],
          [
            /*index*/
            ctx[122],
            /*j*/
            ctx[127]
          ]
        );
      if (dirty[0] & /*datatype*/
      1 | dirty[3] & /*item*/
      1073741824)
        editablecell_changes.datatype = Array.isArray(
          /*datatype*/
          ctx[0]
        ) ? (
          /*datatype*/
          ctx[0][
            /*j*/
            ctx[127]
          ]
        ) : (
          /*datatype*/
          ctx[0]
        );
      if (dirty[0] & /*clear_on_focus*/
      1073741824)
        editablecell_changes.clear_on_focus = /*clear_on_focus*/
        ctx[30];
      if (dirty[0] & /*root*/
      256)
        editablecell_changes.root = /*root*/
        ctx[8];
      if (!updating_value && dirty[0] & /*data*/
      4194304 | dirty[3] & /*index, item*/
      1610612736) {
        updating_value = true;
        editablecell_changes.value = /*data*/
        ctx[22][
          /*index*/
          ctx[122]
        ][
          /*j*/
          ctx[127]
        ].value;
        add_flush_callback(() => updating_value = false);
      }
      if (!updating_el && dirty[0] & /*els*/
      536870912 | dirty[3] & /*item*/
      1073741824) {
        updating_el = true;
        editablecell_changes.el = /*els*/
        ctx[29][
          /*id*/
          ctx[125]
        ].input;
        add_flush_callback(() => updating_el = false);
      }
      editablecell.$set(editablecell_changes);
      if (
        /*editable*/
        ctx[6]
      ) {
        if (if_block) {
          if_block.p(ctx, dirty);
        } else {
          if_block = create_if_block_2(ctx);
          if_block.c();
          if_block.m(div, null);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (!current || dirty[0] & /*styling*/
      524288 | dirty[3] & /*index, item*/
      1610612736 && td_style_value !== (td_style_value = /*styling*/
      ((_d = (_c = ctx[19]) == null ? void 0 : _c[
        /*index*/
        ctx[122]
      ]) == null ? void 0 : _d[
        /*j*/
        ctx[127]
      ]) || "")) {
        attr(td, "style", td_style_value);
      }
      if (!current || dirty[0] & /*selected*/
      1048576 | dirty[3] & /*index, item*/
      1610612736) {
        toggle_class(td, "focus", dequal(
          /*selected*/
          ctx[20],
          [
            /*index*/
            ctx[122],
            /*j*/
            ctx[127]
          ]
        ));
      }
      if (!current || dirty[1] & /*active_cell_menu*/
      128 | dirty[3] & /*index, item*/
      1610612736) {
        toggle_class(
          td,
          "menu-active",
          /*active_cell_menu*/
          ctx[38] && /*active_cell_menu*/
          ctx[38].row === /*index*/
          ctx[122] && /*active_cell_menu*/
          ctx[38].col === /*j*/
          ctx[127]
        );
      }
      const style_changed = dirty[0] & /*styling*/
      524288 | dirty[3] & /*index, item*/
      1610612736;
      if (dirty[0] & /*styling*/
      524288 | dirty[3] & /*item, index*/
      1610612736 && style_width !== (style_width = `var(--cell-width-${/*j*/
      ctx[127]})`) || style_changed) {
        set_style(td, "width", style_width);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(editablecell.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(editablecell.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(td);
      }
      destroy_component(editablecell);
      if (if_block)
        if_block.d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_tbody_slot(ctx) {
  let tr;
  let t;
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let current;
  let if_block = (
    /*show_row_numbers*/
    ctx[13] && create_if_block_3(ctx)
  );
  let each_value = ensure_array_like(
    /*item*/
    ctx[123]
  );
  const get_key = (ctx2) => (
    /*id*/
    ctx2[125]
  );
  for (let i = 0; i < each_value.length; i += 1) {
    let child_ctx = get_each_context(ctx, each_value, i);
    let key = get_key(child_ctx);
    each_1_lookup.set(key, each_blocks[i] = create_each_block(key, child_ctx));
  }
  return {
    c() {
      tr = element("tr");
      if (if_block)
        if_block.c();
      t = space();
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      this.h();
    },
    l(nodes) {
      tr = claim_element(nodes, "TR", { slot: true, class: true });
      var tr_nodes = children(tr);
      if (if_block)
        if_block.l(tr_nodes);
      t = claim_space(tr_nodes);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(tr_nodes);
      }
      tr_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(tr, "slot", "tbody");
      attr(tr, "class", "svelte-1l6bino");
      toggle_class(
        tr,
        "row_odd",
        /*index*/
        ctx[122] % 2 === 0
      );
    },
    m(target, anchor) {
      insert_hydration(target, tr, anchor);
      if (if_block)
        if_block.m(tr, null);
      append_hydration(tr, t);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(tr, null);
        }
      }
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*show_row_numbers*/
        ctx2[13]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block_3(ctx2);
          if_block.c();
          if_block.m(tr, t);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (dirty[0] & /*styling, selected, clear_on_focus, clicked_cell, editable, editing, display_value, latex_delimiters, line_breaks, datatype, root, data, els*/
      1952188769 | dirty[1] & /*active_cell_menu, selected_header, header_edit, toggle_cell_button, active_header_menu, toggle_cell_menu, parent*/
      8913291 | dirty[3] & /*index, item*/
      1610612736) {
        each_value = ensure_array_like(
          /*item*/
          ctx2[123]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value, each_1_lookup, tr, outro_and_destroy_block, create_each_block, null, get_each_context);
        check_outros();
      }
      if (!current || dirty[3] & /*index*/
      536870912) {
        toggle_class(
          tr,
          "row_odd",
          /*index*/
          ctx2[122] % 2 === 0
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
        detach(tr);
      }
      if (if_block)
        if_block.d();
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d();
      }
    }
  };
}
function create_default_slot$1(ctx) {
  let virtualtable;
  let updating_items;
  let updating_actual_height;
  let updating_table_scrollbar_width;
  let current;
  function virtualtable_items_binding(value) {
    ctx[82](value);
  }
  function virtualtable_actual_height_binding(value) {
    ctx[83](value);
  }
  function virtualtable_table_scrollbar_width_binding(value) {
    ctx[84](value);
  }
  let virtualtable_props = {
    max_height: (
      /*max_height*/
      ctx[10]
    ),
    selected: (
      /*selected_index*/
      ctx[41]
    ),
    $$slots: {
      tbody: [
        create_tbody_slot,
        ({ index, item }) => ({ 122: index, 123: item }),
        ({ index, item }) => [0, 0, 0, (index ? 536870912 : 0) | (item ? 1073741824 : 0)]
      ],
      thead: [create_thead_slot],
      default: [create_default_slot_1]
    },
    $$scope: { ctx }
  };
  if (
    /*data*/
    ctx[22] !== void 0
  ) {
    virtualtable_props.items = /*data*/
    ctx[22];
  }
  if (
    /*table_height*/
    ctx[36] !== void 0
  ) {
    virtualtable_props.actual_height = /*table_height*/
    ctx[36];
  }
  if (
    /*scrollbar_width*/
    ctx[37] !== void 0
  ) {
    virtualtable_props.table_scrollbar_width = /*scrollbar_width*/
    ctx[37];
  }
  virtualtable = new VirtualTable({ props: virtualtable_props });
  binding_callbacks.push(() => bind(virtualtable, "items", virtualtable_items_binding));
  binding_callbacks.push(() => bind(virtualtable, "actual_height", virtualtable_actual_height_binding));
  binding_callbacks.push(() => bind(virtualtable, "table_scrollbar_width", virtualtable_table_scrollbar_width_binding));
  return {
    c() {
      create_component(virtualtable.$$.fragment);
    },
    l(nodes) {
      claim_component(virtualtable.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(virtualtable, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const virtualtable_changes = {};
      if (dirty[0] & /*max_height*/
      1024)
        virtualtable_changes.max_height = /*max_height*/
        ctx2[10];
      if (dirty[1] & /*selected_index*/
      1024)
        virtualtable_changes.selected = /*selected_index*/
        ctx2[41];
      if (dirty[0] & /*styling, selected, clear_on_focus, clicked_cell, editable, editing, display_value, latex_delimiters, line_breaks, datatype, root, data, els, show_row_numbers, _headers, sort_by, sort_direction, label*/
      1979459939 | dirty[1] & /*active_cell_menu, selected_header, header_edit, active_header_menu, parent*/
      395 | dirty[3] & /*index, item*/
      1610612736 | dirty[4] & /*$$scope*/
      256) {
        virtualtable_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_items && dirty[0] & /*data*/
      4194304) {
        updating_items = true;
        virtualtable_changes.items = /*data*/
        ctx2[22];
        add_flush_callback(() => updating_items = false);
      }
      if (!updating_actual_height && dirty[1] & /*table_height*/
      32) {
        updating_actual_height = true;
        virtualtable_changes.actual_height = /*table_height*/
        ctx2[36];
        add_flush_callback(() => updating_actual_height = false);
      }
      if (!updating_table_scrollbar_width && dirty[1] & /*scrollbar_width*/
      64) {
        updating_table_scrollbar_width = true;
        virtualtable_changes.table_scrollbar_width = /*scrollbar_width*/
        ctx2[37];
        add_flush_callback(() => updating_table_scrollbar_width = false);
      }
      virtualtable.$set(virtualtable_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(virtualtable.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(virtualtable.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(virtualtable, detaching);
    }
  };
}
function create_if_block_1(ctx) {
  let cellmenu;
  let current;
  cellmenu = new CellMenu({
    props: {
      x: (
        /*active_cell_menu*/
        ctx[38].x
      ),
      y: (
        /*active_cell_menu*/
        ctx[38].y
      ),
      row: (
        /*active_cell_menu*/
        ctx[38].row
      ),
      col_count: (
        /*col_count*/
        ctx[3]
      ),
      row_count: (
        /*row_count*/
        ctx[4]
      ),
      on_add_row_above: (
        /*func*/
        ctx[89]
      ),
      on_add_row_below: (
        /*func_1*/
        ctx[90]
      ),
      on_add_column_left: (
        /*func_2*/
        ctx[91]
      ),
      on_add_column_right: (
        /*func_3*/
        ctx[92]
      ),
      on_delete_row: (
        /*func_4*/
        ctx[93]
      ),
      on_delete_col: (
        /*func_5*/
        ctx[94]
      ),
      can_delete_rows: (
        /*data*/
        ctx[22].length > 1
      ),
      can_delete_cols: (
        /*data*/
        ctx[22][0].length > 1
      ),
      i18n: (
        /*i18n*/
        ctx[9]
      )
    }
  });
  return {
    c() {
      create_component(cellmenu.$$.fragment);
    },
    l(nodes) {
      claim_component(cellmenu.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(cellmenu, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const cellmenu_changes = {};
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.x = /*active_cell_menu*/
        ctx2[38].x;
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.y = /*active_cell_menu*/
        ctx2[38].y;
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.row = /*active_cell_menu*/
        ctx2[38].row;
      if (dirty[0] & /*col_count*/
      8)
        cellmenu_changes.col_count = /*col_count*/
        ctx2[3];
      if (dirty[0] & /*row_count*/
      16)
        cellmenu_changes.row_count = /*row_count*/
        ctx2[4];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_row_above = /*func*/
        ctx2[89];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_row_below = /*func_1*/
        ctx2[90];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_column_left = /*func_2*/
        ctx2[91];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_column_right = /*func_3*/
        ctx2[92];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_delete_row = /*func_4*/
        ctx2[93];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_delete_col = /*func_5*/
        ctx2[94];
      if (dirty[0] & /*data*/
      4194304)
        cellmenu_changes.can_delete_rows = /*data*/
        ctx2[22].length > 1;
      if (dirty[0] & /*data*/
      4194304)
        cellmenu_changes.can_delete_cols = /*data*/
        ctx2[22][0].length > 1;
      if (dirty[0] & /*i18n*/
      512)
        cellmenu_changes.i18n = /*i18n*/
        ctx2[9];
      cellmenu.$set(cellmenu_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenu.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(cellmenu.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(cellmenu, detaching);
    }
  };
}
function create_if_block(ctx) {
  let cellmenu;
  let current;
  cellmenu = new CellMenu({
    props: {
      i18n: (
        /*i18n*/
        ctx[9]
      ),
      x: (
        /*active_header_menu*/
        ctx[39].x
      ),
      y: (
        /*active_header_menu*/
        ctx[39].y
      ),
      row: -1,
      col_count: (
        /*col_count*/
        ctx[3]
      ),
      row_count: (
        /*row_count*/
        ctx[4]
      ),
      on_add_row_above: (
        /*func_6*/
        ctx[95]
      ),
      on_add_row_below: (
        /*func_7*/
        ctx[96]
      ),
      on_add_column_left: (
        /*func_8*/
        ctx[97]
      ),
      on_add_column_right: (
        /*func_9*/
        ctx[98]
      ),
      on_delete_row: (
        /*func_10*/
        ctx[99]
      ),
      on_delete_col: (
        /*func_11*/
        ctx[100]
      ),
      can_delete_rows: false,
      can_delete_cols: (
        /*data*/
        ctx[22][0].length > 1
      )
    }
  });
  return {
    c() {
      create_component(cellmenu.$$.fragment);
    },
    l(nodes) {
      claim_component(cellmenu.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(cellmenu, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const cellmenu_changes = {};
      if (dirty[0] & /*i18n*/
      512)
        cellmenu_changes.i18n = /*i18n*/
        ctx2[9];
      if (dirty[1] & /*active_header_menu*/
      256)
        cellmenu_changes.x = /*active_header_menu*/
        ctx2[39].x;
      if (dirty[1] & /*active_header_menu*/
      256)
        cellmenu_changes.y = /*active_header_menu*/
        ctx2[39].y;
      if (dirty[0] & /*col_count*/
      8)
        cellmenu_changes.col_count = /*col_count*/
        ctx2[3];
      if (dirty[0] & /*row_count*/
      16)
        cellmenu_changes.row_count = /*row_count*/
        ctx2[4];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_row_above = /*func_6*/
        ctx2[95];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_add_row_below = /*func_7*/
        ctx2[96];
      if (dirty[1] & /*active_header_menu*/
      256)
        cellmenu_changes.on_add_column_left = /*func_8*/
        ctx2[97];
      if (dirty[1] & /*active_header_menu*/
      256)
        cellmenu_changes.on_add_column_right = /*func_9*/
        ctx2[98];
      if (dirty[1] & /*active_cell_menu*/
      128)
        cellmenu_changes.on_delete_row = /*func_10*/
        ctx2[99];
      if (dirty[1] & /*active_header_menu*/
      256)
        cellmenu_changes.on_delete_col = /*func_11*/
        ctx2[100];
      if (dirty[0] & /*data*/
      4194304)
        cellmenu_changes.can_delete_cols = /*data*/
        ctx2[22][0].length > 1;
      cellmenu.$set(cellmenu_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(cellmenu.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(cellmenu.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(cellmenu, detaching);
    }
  };
}
function create_fragment$1(ctx) {
  let div2;
  let div0;
  let t0;
  let toolbar;
  let t1;
  let div1;
  let table_1;
  let t2;
  let thead;
  let tr0;
  let t3;
  let each_blocks_1 = [];
  let each0_lookup = /* @__PURE__ */ new Map();
  let t4;
  let tbody;
  let tr1;
  let each_blocks = [];
  let each1_lookup = /* @__PURE__ */ new Map();
  let table_1_resize_listener;
  let t5;
  let upload_1;
  let updating_dragging;
  let t6;
  let t7;
  let if_block4_anchor;
  let current;
  let mounted;
  let dispose;
  let if_block0 = (
    /*label*/
    ctx[1] && /*label*/
    ctx[1].length !== 0 && /*show_label*/
    ctx[2] && create_if_block_9(ctx)
  );
  toolbar = new Toolbar({
    props: {
      show_fullscreen_button: (
        /*show_fullscreen_button*/
        ctx[16]
      ),
      is_fullscreen: (
        /*is_fullscreen*/
        ctx[40]
      ),
      on_copy: (
        /*handle_copy*/
        ctx[56]
      ),
      show_copy_button: (
        /*show_copy_button*/
        ctx[17]
      )
    }
  });
  toolbar.$on(
    "click",
    /*toggle_fullscreen*/
    ctx[55]
  );
  let if_block1 = (
    /*label*/
    ctx[1] && /*label*/
    ctx[1].length !== 0 && create_if_block_8(ctx)
  );
  let if_block2 = (
    /*show_row_numbers*/
    ctx[13] && create_if_block_7()
  );
  let each_value_3 = ensure_array_like(
    /*_headers*/
    ctx[21]
  );
  const get_key = (ctx2) => (
    /*id*/
    ctx2[125]
  );
  for (let i = 0; i < each_value_3.length; i += 1) {
    let child_ctx = get_each_context_3(ctx, each_value_3, i);
    let key = get_key(child_ctx);
    each0_lookup.set(key, each_blocks_1[i] = create_each_block_3(key, child_ctx));
  }
  let each_value_2 = ensure_array_like(
    /*max*/
    ctx[42]
  );
  const get_key_1 = (ctx2) => (
    /*id*/
    ctx2[125]
  );
  for (let i = 0; i < each_value_2.length; i += 1) {
    let child_ctx = get_each_context_2(ctx, each_value_2, i);
    let key = get_key_1(child_ctx);
    each1_lookup.set(key, each_blocks[i] = create_each_block_2(key, child_ctx));
  }
  function upload_1_dragging_binding(value) {
    ctx[85](value);
  }
  let upload_1_props = {
    upload: (
      /*upload*/
      ctx[14]
    ),
    stream_handler: (
      /*stream_handler*/
      ctx[15]
    ),
    flex: false,
    center: false,
    boundedheight: false,
    disable_click: true,
    root: (
      /*root*/
      ctx[8]
    ),
    $$slots: { default: [create_default_slot$1] },
    $$scope: { ctx }
  };
  if (
    /*dragging*/
    ctx[33] !== void 0
  ) {
    upload_1_props.dragging = /*dragging*/
    ctx[33];
  }
  upload_1 = new Upload({ props: upload_1_props });
  binding_callbacks.push(() => bind(upload_1, "dragging", upload_1_dragging_binding));
  upload_1.$on(
    "load",
    /*load_handler*/
    ctx[86]
  );
  let if_block3 = (
    /*active_cell_menu*/
    ctx[38] && create_if_block_1(ctx)
  );
  let if_block4 = (
    /*active_header_menu*/
    ctx[39] !== null && create_if_block(ctx)
  );
  return {
    c() {
      div2 = element("div");
      div0 = element("div");
      if (if_block0)
        if_block0.c();
      t0 = space();
      create_component(toolbar.$$.fragment);
      t1 = space();
      div1 = element("div");
      table_1 = element("table");
      if (if_block1)
        if_block1.c();
      t2 = space();
      thead = element("thead");
      tr0 = element("tr");
      if (if_block2)
        if_block2.c();
      t3 = space();
      for (let i = 0; i < each_blocks_1.length; i += 1) {
        each_blocks_1[i].c();
      }
      t4 = space();
      tbody = element("tbody");
      tr1 = element("tr");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      t5 = space();
      create_component(upload_1.$$.fragment);
      t6 = space();
      if (if_block3)
        if_block3.c();
      t7 = space();
      if (if_block4)
        if_block4.c();
      if_block4_anchor = empty();
      this.h();
    },
    l(nodes) {
      div2 = claim_element(nodes, "DIV", { class: true });
      var div2_nodes = children(div2);
      div0 = claim_element(div2_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      if (if_block0)
        if_block0.l(div0_nodes);
      t0 = claim_space(div0_nodes);
      claim_component(toolbar.$$.fragment, div0_nodes);
      div0_nodes.forEach(detach);
      t1 = claim_space(div2_nodes);
      div1 = claim_element(div2_nodes, "DIV", {
        class: true,
        style: true,
        role: true,
        tabindex: true
      });
      var div1_nodes = children(div1);
      table_1 = claim_element(div1_nodes, "TABLE", { class: true });
      var table_1_nodes = children(table_1);
      if (if_block1)
        if_block1.l(table_1_nodes);
      t2 = claim_space(table_1_nodes);
      thead = claim_element(table_1_nodes, "THEAD", { class: true });
      var thead_nodes = children(thead);
      tr0 = claim_element(thead_nodes, "TR", { class: true });
      var tr0_nodes = children(tr0);
      if (if_block2)
        if_block2.l(tr0_nodes);
      t3 = claim_space(tr0_nodes);
      for (let i = 0; i < each_blocks_1.length; i += 1) {
        each_blocks_1[i].l(tr0_nodes);
      }
      tr0_nodes.forEach(detach);
      thead_nodes.forEach(detach);
      t4 = claim_space(table_1_nodes);
      tbody = claim_element(table_1_nodes, "TBODY", {});
      var tbody_nodes = children(tbody);
      tr1 = claim_element(tbody_nodes, "TR", { class: true });
      var tr1_nodes = children(tr1);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(tr1_nodes);
      }
      tr1_nodes.forEach(detach);
      tbody_nodes.forEach(detach);
      table_1_nodes.forEach(detach);
      t5 = claim_space(div1_nodes);
      claim_component(upload_1.$$.fragment, div1_nodes);
      div1_nodes.forEach(detach);
      div2_nodes.forEach(detach);
      t6 = claim_space(nodes);
      if (if_block3)
        if_block3.l(nodes);
      t7 = claim_space(nodes);
      if (if_block4)
        if_block4.l(nodes);
      if_block4_anchor = empty();
      this.h();
    },
    h() {
      attr(div0, "class", "header-row svelte-1l6bino");
      attr(tr0, "class", "svelte-1l6bino");
      attr(thead, "class", "svelte-1l6bino");
      attr(tr1, "class", "svelte-1l6bino");
      attr(table_1, "class", "svelte-1l6bino");
      toggle_class(
        table_1,
        "fixed-layout",
        /*column_widths*/
        ctx[12].length != 0
      );
      attr(div1, "class", "table-wrap svelte-1l6bino");
      set_style(
        div1,
        "height",
        /*table_height*/
        ctx[36] + "px"
      );
      attr(div1, "role", "grid");
      attr(div1, "tabindex", "0");
      toggle_class(
        div1,
        "dragging",
        /*dragging*/
        ctx[33]
      );
      toggle_class(div1, "no-wrap", !/*wrap*/
      ctx[7]);
      attr(div2, "class", "table-container svelte-1l6bino");
    },
    m(target, anchor) {
      insert_hydration(target, div2, anchor);
      append_hydration(div2, div0);
      if (if_block0)
        if_block0.m(div0, null);
      append_hydration(div0, t0);
      mount_component(toolbar, div0, null);
      append_hydration(div2, t1);
      append_hydration(div2, div1);
      append_hydration(div1, table_1);
      if (if_block1)
        if_block1.m(table_1, null);
      append_hydration(table_1, t2);
      append_hydration(table_1, thead);
      append_hydration(thead, tr0);
      if (if_block2)
        if_block2.m(tr0, null);
      append_hydration(tr0, t3);
      for (let i = 0; i < each_blocks_1.length; i += 1) {
        if (each_blocks_1[i]) {
          each_blocks_1[i].m(tr0, null);
        }
      }
      append_hydration(table_1, t4);
      append_hydration(table_1, tbody);
      append_hydration(tbody, tr1);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(tr1, null);
        }
      }
      table_1_resize_listener = resize_observer_content_box.observe(
        table_1,
        /*table_1_elementresizecontentbox_handler*/
        ctx[68].bind(table_1)
      );
      ctx[69](table_1);
      append_hydration(div1, t5);
      mount_component(upload_1, div1, null);
      ctx[87](div1);
      insert_hydration(target, t6, anchor);
      if (if_block3)
        if_block3.m(target, anchor);
      insert_hydration(target, t7, anchor);
      if (if_block4)
        if_block4.m(target, anchor);
      insert_hydration(target, if_block4_anchor, anchor);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            window_1,
            "resize",
            /*resize_handler*/
            ctx[66]
          ),
          listen(
            div1,
            "keydown",
            /*keydown_handler*/
            ctx[88]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (
        /*label*/
        ctx2[1] && /*label*/
        ctx2[1].length !== 0 && /*show_label*/
        ctx2[2]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
        } else {
          if_block0 = create_if_block_9(ctx2);
          if_block0.c();
          if_block0.m(div0, t0);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      const toolbar_changes = {};
      if (dirty[0] & /*show_fullscreen_button*/
      65536)
        toolbar_changes.show_fullscreen_button = /*show_fullscreen_button*/
        ctx2[16];
      if (dirty[1] & /*is_fullscreen*/
      512)
        toolbar_changes.is_fullscreen = /*is_fullscreen*/
        ctx2[40];
      if (dirty[0] & /*show_copy_button*/
      131072)
        toolbar_changes.show_copy_button = /*show_copy_button*/
        ctx2[17];
      toolbar.$set(toolbar_changes);
      if (
        /*label*/
        ctx2[1] && /*label*/
        ctx2[1].length !== 0
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
        } else {
          if_block1 = create_if_block_8(ctx2);
          if_block1.c();
          if_block1.m(table_1, t2);
        }
      } else if (if_block1) {
        if_block1.d(1);
        if_block1 = null;
      }
      if (
        /*show_row_numbers*/
        ctx2[13]
      ) {
        if (if_block2)
          ;
        else {
          if_block2 = create_if_block_7();
          if_block2.c();
          if_block2.m(tr0, t3);
        }
      } else if (if_block2) {
        if_block2.d(1);
        if_block2 = null;
      }
      if (dirty[0] & /*_headers, sort_by, sort_direction, column_widths, latex_delimiters, line_breaks, root*/
      27269408 | dirty[1] & /*get_sort_status, header_edit*/
      4097) {
        each_value_3 = ensure_array_like(
          /*_headers*/
          ctx2[21]
        );
        group_outros();
        each_blocks_1 = update_keyed_each(each_blocks_1, dirty, get_key, 1, ctx2, each_value_3, each0_lookup, tr0, outro_and_destroy_block, create_each_block_3, null, get_each_context_3);
        check_outros();
      }
      if (dirty[0] & /*cells, latex_delimiters, line_breaks, datatype, root*/
      33556769 | dirty[1] & /*max*/
      2048) {
        each_value_2 = ensure_array_like(
          /*max*/
          ctx2[42]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key_1, 1, ctx2, each_value_2, each1_lookup, tr1, outro_and_destroy_block, create_each_block_2, null, get_each_context_2);
        check_outros();
      }
      if (!current || dirty[0] & /*column_widths*/
      4096) {
        toggle_class(
          table_1,
          "fixed-layout",
          /*column_widths*/
          ctx2[12].length != 0
        );
      }
      const upload_1_changes = {};
      if (dirty[0] & /*upload*/
      16384)
        upload_1_changes.upload = /*upload*/
        ctx2[14];
      if (dirty[0] & /*stream_handler*/
      32768)
        upload_1_changes.stream_handler = /*stream_handler*/
        ctx2[15];
      if (dirty[0] & /*root*/
      256)
        upload_1_changes.root = /*root*/
        ctx2[8];
      if (dirty[0] & /*max_height, data, styling, selected, clear_on_focus, clicked_cell, editable, editing, display_value, latex_delimiters, line_breaks, datatype, root, els, show_row_numbers, _headers, sort_by, sort_direction, label*/
      1979460963 | dirty[1] & /*selected_index, table_height, scrollbar_width, active_cell_menu, selected_header, header_edit, active_header_menu, parent*/
      1515 | dirty[4] & /*$$scope*/
      256) {
        upload_1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_dragging && dirty[1] & /*dragging*/
      4) {
        updating_dragging = true;
        upload_1_changes.dragging = /*dragging*/
        ctx2[33];
        add_flush_callback(() => updating_dragging = false);
      }
      upload_1.$set(upload_1_changes);
      if (!current || dirty[1] & /*table_height*/
      32) {
        set_style(
          div1,
          "height",
          /*table_height*/
          ctx2[36] + "px"
        );
      }
      if (!current || dirty[1] & /*dragging*/
      4) {
        toggle_class(
          div1,
          "dragging",
          /*dragging*/
          ctx2[33]
        );
      }
      if (!current || dirty[0] & /*wrap*/
      128) {
        toggle_class(div1, "no-wrap", !/*wrap*/
        ctx2[7]);
      }
      if (
        /*active_cell_menu*/
        ctx2[38]
      ) {
        if (if_block3) {
          if_block3.p(ctx2, dirty);
          if (dirty[1] & /*active_cell_menu*/
          128) {
            transition_in(if_block3, 1);
          }
        } else {
          if_block3 = create_if_block_1(ctx2);
          if_block3.c();
          transition_in(if_block3, 1);
          if_block3.m(t7.parentNode, t7);
        }
      } else if (if_block3) {
        group_outros();
        transition_out(if_block3, 1, 1, () => {
          if_block3 = null;
        });
        check_outros();
      }
      if (
        /*active_header_menu*/
        ctx2[39] !== null
      ) {
        if (if_block4) {
          if_block4.p(ctx2, dirty);
          if (dirty[1] & /*active_header_menu*/
          256) {
            transition_in(if_block4, 1);
          }
        } else {
          if_block4 = create_if_block(ctx2);
          if_block4.c();
          transition_in(if_block4, 1);
          if_block4.m(if_block4_anchor.parentNode, if_block4_anchor);
        }
      } else if (if_block4) {
        group_outros();
        transition_out(if_block4, 1, 1, () => {
          if_block4 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(toolbar.$$.fragment, local);
      for (let i = 0; i < each_value_3.length; i += 1) {
        transition_in(each_blocks_1[i]);
      }
      for (let i = 0; i < each_value_2.length; i += 1) {
        transition_in(each_blocks[i]);
      }
      transition_in(upload_1.$$.fragment, local);
      transition_in(if_block3);
      transition_in(if_block4);
      current = true;
    },
    o(local) {
      transition_out(toolbar.$$.fragment, local);
      for (let i = 0; i < each_blocks_1.length; i += 1) {
        transition_out(each_blocks_1[i]);
      }
      for (let i = 0; i < each_blocks.length; i += 1) {
        transition_out(each_blocks[i]);
      }
      transition_out(upload_1.$$.fragment, local);
      transition_out(if_block3);
      transition_out(if_block4);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div2);
        detach(t6);
        detach(t7);
        detach(if_block4_anchor);
      }
      if (if_block0)
        if_block0.d();
      destroy_component(toolbar);
      if (if_block1)
        if_block1.d();
      if (if_block2)
        if_block2.d();
      for (let i = 0; i < each_blocks_1.length; i += 1) {
        each_blocks_1[i].d();
      }
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d();
      }
      table_1_resize_listener();
      ctx[69](null);
      destroy_component(upload_1);
      ctx[87](null);
      if (if_block3)
        if_block3.d(detaching);
      if (if_block4)
        if_block4.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function make_id() {
  return Math.random().toString(36).substring(2, 15);
}
function guess_delimitaor(text2, possibleDelimiters) {
  return possibleDelimiters.filter(weedOut);
  function weedOut(delimiter) {
    var cache = -1;
    return text2.split("\n").every(checkLength);
    function checkLength(line) {
      if (!line) {
        return true;
      }
      var length = line.split(delimiter).length;
      if (cache < 0) {
        cache = length;
      }
      return cache === length && length > 1;
    }
  }
}
function data_uri_to_blob(data_uri) {
  const byte_str = atob(data_uri.split(",")[1]);
  const mime_str = data_uri.split(",")[0].split(":")[1].split(";")[0];
  const ab = new ArrayBuffer(byte_str.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byte_str.length; i++) {
    ia[i] = byte_str.charCodeAt(i);
  }
  return new Blob([ab], { type: mime_str });
}
function get_max(_d) {
  if (!_d || _d.length === 0 || !_d[0])
    return [];
  let max2 = _d[0].slice();
  for (let i = 0; i < _d.length; i++) {
    for (let j = 0; j < _d[i].length; j++) {
      if (`${max2[j].value}`.length < `${_d[i][j].value}`.length) {
        max2[j] = _d[i][j];
      }
    }
  }
  return max2;
}
const mousedown_handler = (event) => {
  event.preventDefault();
  event.stopPropagation();
};
function instance$1($$self, $$props, $$invalidate) {
  let max;
  let selected_index;
  let { datatype } = $$props;
  let { label = null } = $$props;
  let { show_label = true } = $$props;
  let { headers = [] } = $$props;
  let { values = [] } = $$props;
  let { col_count } = $$props;
  let { row_count } = $$props;
  let { latex_delimiters } = $$props;
  let { editable = true } = $$props;
  let { wrap = false } = $$props;
  let { root } = $$props;
  let { i18n } = $$props;
  let { max_height = 500 } = $$props;
  let { line_breaks = true } = $$props;
  let { column_widths = [] } = $$props;
  let { show_row_numbers = false } = $$props;
  let { upload } = $$props;
  let { stream_handler } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let { show_copy_button = false } = $$props;
  let { value_is_output = false } = $$props;
  let selected = false;
  let clicked_cell = void 0;
  let { display_value = null } = $$props;
  let { styling = null } = $$props;
  let t_rect;
  const dispatch = createEventDispatcher();
  let editing = false;
  const get_data_at = (row, col) => {
    var _a, _b;
    return (_b = (_a = data == null ? void 0 : data[row]) == null ? void 0 : _a[col]) == null ? void 0 : _b.value;
  };
  let last_selected = null;
  let els = {};
  function make_headers(_head) {
    let _h = _head || [];
    if (col_count[1] === "fixed" && _h.length < col_count[0]) {
      const fill = Array(col_count[0] - _h.length).fill("").map((_, i) => `${i + _h.length}`);
      _h = _h.concat(fill);
    }
    if (!_h || _h.length === 0) {
      return Array(col_count[0]).fill(0).map((_, i) => {
        const _id = make_id();
        $$invalidate(29, els[_id] = { cell: null, input: null }, els);
        return { id: _id, value: JSON.stringify(i + 1) };
      });
    }
    return _h.map((h, i) => {
      const _id = make_id();
      $$invalidate(29, els[_id] = { cell: null, input: null }, els);
      return { id: _id, value: h ?? "" };
    });
  }
  function process_data(_values) {
    const data_row_length = _values.length;
    return Array(row_count[1] === "fixed" ? row_count[0] : data_row_length).fill(0).map((_, i) => Array(col_count[1] === "fixed" ? col_count[0] : data_row_length > 0 ? _values[0].length : headers.length).fill(0).map((_2, j) => {
      var _a;
      const id = make_id();
      $$invalidate(29, els[id] = els[id] || { input: null, cell: null }, els);
      const obj = { value: ((_a = _values == null ? void 0 : _values[i]) == null ? void 0 : _a[j]) ?? "", id };
      return obj;
    }));
  }
  let _headers = make_headers(headers);
  let old_headers = headers;
  let data = [[]];
  let old_val = void 0;
  let previous_headers = _headers.map((h) => h.value);
  let previous_data = data.map((row) => row.map((cell) => String(cell.value)));
  async function trigger_change() {
    const current_headers = _headers.map((h) => h.value);
    const current_data = data.map((row) => row.map((cell) => String(cell.value)));
    if (!dequal(current_data, previous_data) || !dequal(current_headers, previous_headers)) {
      dispatch("change", {
        data: data.map((row) => row.map((cell) => cell.value)),
        headers: _headers.map((h) => h.value),
        metadata: null
      });
      if (!value_is_output) {
        dispatch("input");
      }
      previous_data = current_data;
      previous_headers = current_headers;
    }
  }
  function get_sort_status(name, _sort, direction) {
    if (!_sort)
      return "none";
    if (headers[_sort] === name) {
      if (direction === "asc")
        return "ascending";
      if (direction === "des")
        return "descending";
    }
    return "none";
  }
  function get_current_indices(id) {
    return data.reduce(
      (acc, arr, i) => {
        const j = arr.reduce((_acc, _data, k) => id === _data.id ? k : _acc, -1);
        return j === -1 ? acc : [i, j];
      },
      [-1, -1]
    );
  }
  function move_cursor(key, current_coords) {
    var _a;
    const dir = {
      ArrowRight: [0, 1],
      ArrowLeft: [0, -1],
      ArrowDown: [1, 0],
      ArrowUp: [-1, 0]
    }[key];
    const i = current_coords[0] + dir[0];
    const j = current_coords[1] + dir[1];
    if (i < 0 && j <= 0) {
      $$invalidate(32, selected_header = j);
      $$invalidate(20, selected = false);
    } else {
      const is_data = (_a = data[i]) == null ? void 0 : _a[j];
      $$invalidate(20, selected = is_data ? [i, j] : selected);
    }
  }
  let clear_on_focus = false;
  async function handle_keydown(event) {
    var _a;
    if (selected_header !== false && header_edit === false) {
      switch (event.key) {
        case "ArrowDown":
          $$invalidate(20, selected = [0, selected_header]);
          $$invalidate(32, selected_header = false);
          return;
        case "ArrowLeft":
          $$invalidate(32, selected_header = selected_header > 0 ? selected_header - 1 : selected_header);
          return;
        case "ArrowRight":
          $$invalidate(32, selected_header = selected_header < _headers.length - 1 ? selected_header + 1 : selected_header);
          return;
        case "Escape":
          event.preventDefault();
          $$invalidate(32, selected_header = false);
          break;
        case "Enter":
          event.preventDefault();
          break;
      }
    }
    if (!selected) {
      return;
    }
    const [i, j] = selected;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowLeft":
      case "ArrowDown":
      case "ArrowUp":
        if (editing)
          break;
        event.preventDefault();
        move_cursor(event.key, [i, j]);
        break;
      case "Escape":
        if (!editable)
          break;
        event.preventDefault();
        $$invalidate(28, editing = false);
        break;
      case "Enter":
        if (!editable)
          break;
        event.preventDefault();
        if (event.shiftKey) {
          add_row(i);
          await tick();
          $$invalidate(20, selected = [i + 1, j]);
        } else {
          if (dequal(editing, [i, j])) {
            const cell_id = data[i][j].id;
            const input_el = els[cell_id].input;
            if (input_el) {
              $$invalidate(22, data[i][j].value = input_el.value, data);
            }
            $$invalidate(28, editing = false);
            await tick();
            $$invalidate(20, selected = [i, j]);
          } else {
            $$invalidate(28, editing = [i, j]);
          }
        }
        break;
      case "Backspace":
        if (!editable)
          break;
        if (!editing) {
          event.preventDefault();
          $$invalidate(22, data[i][j].value = "", data);
        }
        break;
      case "Delete":
        if (!editable)
          break;
        if (!editing) {
          event.preventDefault();
          $$invalidate(22, data[i][j].value = "", data);
        }
        break;
      case "Tab":
        let direction = event.shiftKey ? -1 : 1;
        let is_data_x = data[i][j + direction];
        let is_data_y = (_a = data == null ? void 0 : data[i + direction]) == null ? void 0 : _a[direction > 0 ? 0 : _headers.length - 1];
        if (is_data_x || is_data_y) {
          event.preventDefault();
          $$invalidate(20, selected = is_data_x ? [i, j + direction] : [i + direction, direction > 0 ? 0 : _headers.length - 1]);
        }
        $$invalidate(28, editing = false);
        break;
      default:
        if (!editable)
          break;
        if ((!editing || editing && dequal(editing, [i, j])) && event.key.length === 1) {
          $$invalidate(30, clear_on_focus = true);
          $$invalidate(28, editing = [i, j]);
        }
    }
  }
  let sort_direction;
  let sort_by;
  function handle_sort(col) {
    if (typeof sort_by !== "number" || sort_by !== col) {
      $$invalidate(23, sort_direction = "asc");
      $$invalidate(24, sort_by = col);
    } else {
      if (sort_direction === "asc") {
        $$invalidate(23, sort_direction = "des");
      } else if (sort_direction === "des") {
        $$invalidate(23, sort_direction = "asc");
      }
    }
  }
  let header_edit;
  let selected_header = false;
  async function edit_header(i, _select = false) {
    if (!editable || col_count[1] !== "dynamic" || header_edit === i)
      return;
    $$invalidate(20, selected = false);
    $$invalidate(32, selected_header = i);
    $$invalidate(31, header_edit = i);
  }
  function end_header_edit(event) {
    if (!editable)
      return;
    switch (event.detail.key) {
      case "Escape":
      case "Enter":
      case "Tab":
        event.preventDefault();
        $$invalidate(20, selected = false);
        $$invalidate(32, selected_header = header_edit);
        $$invalidate(31, header_edit = false);
        parent.focus();
        break;
    }
  }
  async function add_row(index) {
    parent.focus();
    if (row_count[1] !== "dynamic")
      return;
    if (data.length === 0) {
      $$invalidate(61, values = [Array(headers.length).fill("")]);
      return;
    }
    const new_row = Array(data[0].length).fill(0).map((_, i) => {
      const _id = make_id();
      $$invalidate(29, els[_id] = { cell: null, input: null }, els);
      return { id: _id, value: "" };
    });
    if (index !== void 0 && index >= 0 && index <= data.length) {
      data.splice(index, 0, new_row);
    } else {
      data.push(new_row);
    }
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    $$invalidate(20, selected = [index !== void 0 ? index : data.length - 1, 0]);
  }
  async function add_col(index) {
    parent.focus();
    if (col_count[1] !== "dynamic")
      return;
    const insert_index = index !== void 0 ? index : data[0].length;
    for (let i = 0; i < data.length; i++) {
      const _id = make_id();
      $$invalidate(29, els[_id] = { cell: null, input: null }, els);
      data[i].splice(insert_index, 0, { id: _id, value: "" });
    }
    headers.splice(insert_index, 0, `Header ${headers.length + 1}`);
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    $$invalidate(60, headers);
    await tick();
    requestAnimationFrame(() => {
      edit_header(insert_index, true);
      const new_w = parent.querySelectorAll("tbody")[1].offsetWidth;
      parent.querySelectorAll("table")[1].scrollTo({ left: new_w });
    });
  }
  function handle_click_outside(event) {
    if (active_cell_menu && !event.target.closest(".cell-menu") || active_header_menu && !event.target.closest(".cell-menu")) {
      $$invalidate(38, active_cell_menu = null);
      $$invalidate(39, active_header_menu = null);
    }
    const [trigger] = event.composedPath();
    if (parent.contains(trigger)) {
      return;
    }
    $$invalidate(26, clicked_cell = void 0);
    $$invalidate(28, editing = false);
    $$invalidate(20, selected = false);
    $$invalidate(31, header_edit = false);
    $$invalidate(32, selected_header = false);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
  }
  function blob_to_string(blob) {
    const reader = new FileReader();
    function handle_read(e) {
      var _a;
      if (!((_a = e == null ? void 0 : e.target) == null ? void 0 : _a.result) || typeof e.target.result !== "string")
        return;
      const [delimiter] = guess_delimitaor(e.target.result, [",", "	"]);
      const [head, ...rest] = dsvFormat(delimiter).parseRows(e.target.result);
      $$invalidate(21, _headers = make_headers(col_count[1] === "fixed" ? head.slice(0, col_count[0]) : head));
      $$invalidate(61, values = rest);
      reader.removeEventListener("loadend", handle_read);
    }
    reader.addEventListener("loadend", handle_read);
    reader.readAsText(blob);
  }
  let dragging = false;
  let cells = [];
  let parent;
  let table;
  function set_cell_widths() {
    const widths = cells.map((el, i) => {
      return (el == null ? void 0 : el.clientWidth) || 0;
    });
    if (widths.length === 0)
      return;
    for (let i = 0; i < widths.length; i++) {
      parent.style.setProperty(`--cell-width-${i}`, `${widths[i] - scrollbar_width / widths.length}px`);
    }
  }
  let table_height = values.slice(0, max_height / values.length * 37).length * 37 + 37;
  let scrollbar_width = 0;
  function sort_data(_data, _display_value, _styling, col, dir) {
    let id = null;
    if (selected && selected[0] in data && selected[1] in data[selected[0]]) {
      id = data[selected[0]][selected[1]].id;
    }
    if (typeof col !== "number" || !dir) {
      return;
    }
    const indices = [...Array(_data.length).keys()];
    if (dir === "asc") {
      indices.sort((i, j) => _data[i][col].value < _data[j][col].value ? -1 : 1);
    } else if (dir === "des") {
      indices.sort((i, j) => _data[i][col].value > _data[j][col].value ? -1 : 1);
    } else {
      return;
    }
    const temp_data = [..._data];
    const temp_display_value = _display_value ? [..._display_value] : null;
    const temp_styling = _styling ? [..._styling] : null;
    indices.forEach((originalIndex, sortedIndex) => {
      _data[sortedIndex] = temp_data[originalIndex];
      if (_display_value && temp_display_value)
        _display_value[sortedIndex] = temp_display_value[originalIndex];
      if (_styling && temp_styling)
        _styling[sortedIndex] = temp_styling[originalIndex];
    });
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    if (id) {
      const [i, j] = get_current_indices(id);
      $$invalidate(20, selected = [i, j]);
    }
  }
  let is_visible = false;
  onMount(() => {
    const observer = new IntersectionObserver((entries, observer2) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !is_visible) {
          set_cell_widths();
          $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
        }
        is_visible = entry.isIntersecting;
      });
    });
    observer.observe(parent);
    document.addEventListener("click", handle_click_outside);
    window.addEventListener("resize", handle_resize);
    document.addEventListener("fullscreenchange", handle_fullscreen_change);
    return () => {
      observer.disconnect();
      document.removeEventListener("click", handle_click_outside);
      window.removeEventListener("resize", handle_resize);
      document.removeEventListener("fullscreenchange", handle_fullscreen_change);
    };
  });
  let active_cell_menu = null;
  function toggle_cell_menu(event, row, col) {
    event.stopPropagation();
    if (active_cell_menu && active_cell_menu.row === row && active_cell_menu.col === col) {
      $$invalidate(38, active_cell_menu = null);
    } else {
      const cell = event.target.closest("td");
      if (cell) {
        const rect = cell.getBoundingClientRect();
        $$invalidate(38, active_cell_menu = { row, col, x: rect.right, y: rect.bottom });
      }
    }
  }
  function add_row_at(index, position) {
    const row_index = position === "above" ? index : index + 1;
    add_row(row_index);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
  }
  function add_col_at(index, position) {
    const col_index = position === "left" ? index : index + 1;
    add_col(col_index);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
  }
  function handle_resize() {
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
    set_cell_widths();
  }
  let active_button = null;
  function toggle_header_button(col) {
    if ((active_button == null ? void 0 : active_button.type) === "header" && active_button.col === col) {
      active_button = null;
    } else {
      active_button = { type: "header", col };
    }
  }
  function toggle_cell_button(row, col) {
    if ((active_button == null ? void 0 : active_button.type) === "cell" && active_button.row === row && active_button.col === col) {
      active_button = null;
    } else {
      active_button = { type: "cell", row, col };
    }
  }
  let active_header_menu = null;
  let is_fullscreen = false;
  function toggle_fullscreen() {
    if (!document.fullscreenElement) {
      parent.requestFullscreen();
      $$invalidate(40, is_fullscreen = true);
    } else {
      document.exitFullscreen();
      $$invalidate(40, is_fullscreen = false);
    }
  }
  function handle_fullscreen_change() {
    $$invalidate(40, is_fullscreen = !!document.fullscreenElement);
  }
  async function handle_copy() {
    await copy_table_data(data, _headers);
  }
  function toggle_header_menu(event, col) {
    event.stopPropagation();
    if (active_header_menu && active_header_menu.col === col) {
      $$invalidate(39, active_header_menu = null);
    } else {
      const header = event.target.closest("th");
      if (header) {
        const rect = header.getBoundingClientRect();
        $$invalidate(39, active_header_menu = { col, x: rect.right, y: rect.bottom });
      }
    }
  }
  afterUpdate(() => {
    $$invalidate(62, value_is_output = false);
  });
  async function delete_row(index) {
    parent.focus();
    if (row_count[1] !== "dynamic")
      return;
    if (data.length <= 1)
      return;
    data.splice(index, 1);
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    $$invalidate(20, selected = false);
  }
  async function delete_col(index) {
    parent.focus();
    if (col_count[1] !== "dynamic")
      return;
    if (data[0].length <= 1)
      return;
    _headers.splice(index, 1);
    $$invalidate(21, _headers), $$invalidate(60, headers), $$invalidate(64, old_headers);
    data.forEach((row) => {
      row.splice(index, 1);
    });
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    $$invalidate(20, selected = false);
  }
  function delete_row_at(index) {
    delete_row(index);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
  }
  function delete_col_at(index) {
    delete_col(index);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
  }
  const resize_handler = () => set_cell_widths();
  function td_binding($$value, j) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      cells[j] = $$value;
      $$invalidate(25, cells);
    });
  }
  function table_1_elementresizecontentbox_handler() {
    var _a;
    t_rect = (_a = ResizeObserverSingleton.entries.get(this)) == null ? void 0 : _a.contentRect;
    $$invalidate(27, t_rect);
  }
  function table_1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      table = $$value;
      $$invalidate(35, table);
    });
  }
  function editablecell_value_binding(value, index, j) {
    if ($$self.$$.not_equal(data[index][j].value, value)) {
      data[index][j].value = value;
      $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
    }
  }
  function editablecell_el_binding(value, id) {
    if ($$self.$$.not_equal(els[id].input, value)) {
      els[id].input = value;
      $$invalidate(29, els);
    }
  }
  const blur_handler = () => {
    $$invalidate(30, clear_on_focus = false);
    parent.focus();
  };
  const click_handler = (index, j, event) => toggle_cell_menu(event, index, j);
  const touchstart_handler = (index, j, event) => {
    event.preventDefault();
    event.stopPropagation();
    $$invalidate(30, clear_on_focus = false);
    $$invalidate(26, clicked_cell = { row: index, col: j });
    $$invalidate(20, selected = [index, j]);
    $$invalidate(32, selected_header = false);
    $$invalidate(31, header_edit = false);
    if (editable) {
      $$invalidate(28, editing = [index, j]);
    }
    toggle_cell_button(index, j);
  };
  const click_handler_1 = (index, j, event) => {
    event.preventDefault();
    event.stopPropagation();
    $$invalidate(30, clear_on_focus = false);
    $$invalidate(38, active_cell_menu = null);
    $$invalidate(39, active_header_menu = null);
    $$invalidate(26, clicked_cell = { row: index, col: j });
    $$invalidate(20, selected = [index, j]);
    $$invalidate(32, selected_header = false);
    $$invalidate(31, header_edit = false);
    if (editable) {
      $$invalidate(28, editing = [index, j]);
    }
    toggle_cell_button(index, j);
  };
  function editablecell_value_binding_1(value, i) {
    if ($$self.$$.not_equal(_headers[i].value, value)) {
      _headers[i].value = value;
      $$invalidate(21, _headers), $$invalidate(60, headers), $$invalidate(64, old_headers);
    }
  }
  function editablecell_el_binding_1(value, id) {
    if ($$self.$$.not_equal(els[id].input, value)) {
      els[id].input = value;
      $$invalidate(29, els);
    }
  }
  const dblclick_handler = (i) => edit_header(i);
  const click_handler_2 = (i, event) => {
    event.stopPropagation();
    handle_sort(i);
  };
  const click_handler_3 = (i, event) => toggle_header_menu(event, i);
  const click_handler_4 = (i) => {
    toggle_header_button(i);
  };
  function virtualtable_items_binding(value) {
    data = value;
    $$invalidate(22, data), $$invalidate(61, values), $$invalidate(65, old_val);
  }
  function virtualtable_actual_height_binding(value) {
    table_height = value;
    $$invalidate(36, table_height);
  }
  function virtualtable_table_scrollbar_width_binding(value) {
    scrollbar_width = value;
    $$invalidate(37, scrollbar_width);
  }
  function upload_1_dragging_binding(value) {
    dragging = value;
    $$invalidate(33, dragging);
  }
  const load_handler = (e) => blob_to_string(data_uri_to_blob(e.detail.data));
  function div1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      parent = $$value;
      $$invalidate(34, parent);
    });
  }
  const keydown_handler = (e) => handle_keydown(e);
  const func = () => add_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) || 0, "above");
  const func_1 = () => add_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) || 0, "below");
  const func_2 = () => add_col_at((active_cell_menu == null ? void 0 : active_cell_menu.col) || 0, "left");
  const func_3 = () => add_col_at((active_cell_menu == null ? void 0 : active_cell_menu.col) || 0, "right");
  const func_4 = () => delete_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) || 0);
  const func_5 = () => delete_col_at((active_cell_menu == null ? void 0 : active_cell_menu.col) || 0);
  const func_6 = () => add_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) ?? -1, "above");
  const func_7 = () => add_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) ?? -1, "below");
  const func_8 = () => add_col_at((active_header_menu == null ? void 0 : active_header_menu.col) ?? -1, "left");
  const func_9 = () => add_col_at((active_header_menu == null ? void 0 : active_header_menu.col) ?? -1, "right");
  const func_10 = () => delete_row_at((active_cell_menu == null ? void 0 : active_cell_menu.row) ?? -1);
  const func_11 = () => delete_col_at((active_header_menu == null ? void 0 : active_header_menu.col) ?? -1);
  $$self.$$set = ($$props2) => {
    if ("datatype" in $$props2)
      $$invalidate(0, datatype = $$props2.datatype);
    if ("label" in $$props2)
      $$invalidate(1, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(2, show_label = $$props2.show_label);
    if ("headers" in $$props2)
      $$invalidate(60, headers = $$props2.headers);
    if ("values" in $$props2)
      $$invalidate(61, values = $$props2.values);
    if ("col_count" in $$props2)
      $$invalidate(3, col_count = $$props2.col_count);
    if ("row_count" in $$props2)
      $$invalidate(4, row_count = $$props2.row_count);
    if ("latex_delimiters" in $$props2)
      $$invalidate(5, latex_delimiters = $$props2.latex_delimiters);
    if ("editable" in $$props2)
      $$invalidate(6, editable = $$props2.editable);
    if ("wrap" in $$props2)
      $$invalidate(7, wrap = $$props2.wrap);
    if ("root" in $$props2)
      $$invalidate(8, root = $$props2.root);
    if ("i18n" in $$props2)
      $$invalidate(9, i18n = $$props2.i18n);
    if ("max_height" in $$props2)
      $$invalidate(10, max_height = $$props2.max_height);
    if ("line_breaks" in $$props2)
      $$invalidate(11, line_breaks = $$props2.line_breaks);
    if ("column_widths" in $$props2)
      $$invalidate(12, column_widths = $$props2.column_widths);
    if ("show_row_numbers" in $$props2)
      $$invalidate(13, show_row_numbers = $$props2.show_row_numbers);
    if ("upload" in $$props2)
      $$invalidate(14, upload = $$props2.upload);
    if ("stream_handler" in $$props2)
      $$invalidate(15, stream_handler = $$props2.stream_handler);
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(16, show_fullscreen_button = $$props2.show_fullscreen_button);
    if ("show_copy_button" in $$props2)
      $$invalidate(17, show_copy_button = $$props2.show_copy_button);
    if ("value_is_output" in $$props2)
      $$invalidate(62, value_is_output = $$props2.value_is_output);
    if ("display_value" in $$props2)
      $$invalidate(18, display_value = $$props2.display_value);
    if ("styling" in $$props2)
      $$invalidate(19, styling = $$props2.styling);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[1] & /*values*/
    1073741824 | $$self.$$.dirty[2] & /*old_val*/
    8) {
      if (!dequal(values, old_val)) {
        $$invalidate(22, data = process_data(values));
        $$invalidate(65, old_val = JSON.parse(JSON.stringify(values)));
      }
    }
    if ($$self.$$.dirty[0] & /*selected, data*/
    5242880 | $$self.$$.dirty[2] & /*last_selected*/
    2) {
      {
        if (selected !== false && !dequal(selected, last_selected)) {
          const [row, col] = selected;
          if (!isNaN(row) && !isNaN(col) && data[row]) {
            dispatch("select", {
              index: [row, col],
              value: get_data_at(row, col),
              row_value: data[row].map((d) => d.value)
            });
            $$invalidate(63, last_selected = selected);
          }
        }
      }
    }
    if ($$self.$$.dirty[1] & /*headers*/
    536870912 | $$self.$$.dirty[2] & /*old_headers*/
    4) {
      {
        if (!dequal(headers, old_headers)) {
          $$invalidate(21, _headers = make_headers(headers));
          $$invalidate(64, old_headers = JSON.parse(JSON.stringify(headers)));
        }
      }
    }
    if ($$self.$$.dirty[0] & /*data, _headers*/
    6291456) {
      (data || _headers) && trigger_change();
    }
    if ($$self.$$.dirty[0] & /*data*/
    4194304) {
      $$invalidate(42, max = get_max(data));
    }
    if ($$self.$$.dirty[0] & /*cells*/
    33554432) {
      cells[0] && set_cell_widths();
    }
    if ($$self.$$.dirty[0] & /*data, display_value, styling, sort_by, sort_direction*/
    30146560) {
      sort_data(data, display_value, styling, sort_by, sort_direction);
    }
    if ($$self.$$.dirty[0] & /*selected*/
    1048576) {
      $$invalidate(41, selected_index = !!selected && selected[0]);
    }
  };
  return [
    datatype,
    label,
    show_label,
    col_count,
    row_count,
    latex_delimiters,
    editable,
    wrap,
    root,
    i18n,
    max_height,
    line_breaks,
    column_widths,
    show_row_numbers,
    upload,
    stream_handler,
    show_fullscreen_button,
    show_copy_button,
    display_value,
    styling,
    selected,
    _headers,
    data,
    sort_direction,
    sort_by,
    cells,
    clicked_cell,
    t_rect,
    editing,
    els,
    clear_on_focus,
    header_edit,
    selected_header,
    dragging,
    parent,
    table,
    table_height,
    scrollbar_width,
    active_cell_menu,
    active_header_menu,
    is_fullscreen,
    selected_index,
    max,
    get_sort_status,
    handle_keydown,
    handle_sort,
    edit_header,
    end_header_edit,
    blob_to_string,
    set_cell_widths,
    toggle_cell_menu,
    add_row_at,
    add_col_at,
    toggle_header_button,
    toggle_cell_button,
    toggle_fullscreen,
    handle_copy,
    toggle_header_menu,
    delete_row_at,
    delete_col_at,
    headers,
    values,
    value_is_output,
    last_selected,
    old_headers,
    old_val,
    resize_handler,
    td_binding,
    table_1_elementresizecontentbox_handler,
    table_1_binding,
    editablecell_value_binding,
    editablecell_el_binding,
    blur_handler,
    click_handler,
    touchstart_handler,
    click_handler_1,
    editablecell_value_binding_1,
    editablecell_el_binding_1,
    dblclick_handler,
    click_handler_2,
    click_handler_3,
    click_handler_4,
    virtualtable_items_binding,
    virtualtable_actual_height_binding,
    virtualtable_table_scrollbar_width_binding,
    upload_1_dragging_binding,
    load_handler,
    div1_binding,
    keydown_handler,
    func,
    func_1,
    func_2,
    func_3,
    func_4,
    func_5,
    func_6,
    func_7,
    func_8,
    func_9,
    func_10,
    func_11
  ];
}
class Table extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance$1,
      create_fragment$1,
      safe_not_equal,
      {
        datatype: 0,
        label: 1,
        show_label: 2,
        headers: 60,
        values: 61,
        col_count: 3,
        row_count: 4,
        latex_delimiters: 5,
        editable: 6,
        wrap: 7,
        root: 8,
        i18n: 9,
        max_height: 10,
        line_breaks: 11,
        column_widths: 12,
        show_row_numbers: 13,
        upload: 14,
        stream_handler: 15,
        show_fullscreen_button: 16,
        show_copy_button: 17,
        value_is_output: 62,
        display_value: 18,
        styling: 19
      },
      null,
      [-1, -1, -1, -1, -1]
    );
  }
}
const Table$1 = Table;
function create_default_slot(ctx) {
  let statustracker;
  let t;
  let table;
  let updating_value_is_output;
  let current;
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[16].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[16].i18n
    ) },
    /*loading_status*/
    ctx[19]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[28]
  );
  function table_value_is_output_binding(value) {
    ctx[31](value);
  }
  let table_props = {
    root: (
      /*root*/
      ctx[13]
    ),
    label: (
      /*label*/
      ctx[7]
    ),
    show_label: (
      /*show_label*/
      ctx[8]
    ),
    row_count: (
      /*row_count*/
      ctx[6]
    ),
    col_count: (
      /*col_count*/
      ctx[5]
    ),
    values: (
      /*cell_values*/
      ctx[25]
    ),
    display_value: (
      /*display_value*/
      ctx[24]
    ),
    styling: (
      /*styling*/
      ctx[23]
    ),
    headers: (
      /*_headers*/
      ctx[26]
    ),
    wrap: (
      /*wrap*/
      ctx[9]
    ),
    datatype: (
      /*datatype*/
      ctx[10]
    ),
    latex_delimiters: (
      /*latex_delimiters*/
      ctx[17]
    ),
    editable: (
      /*interactive*/
      ctx[20]
    ),
    max_height: (
      /*max_height*/
      ctx[18]
    ),
    i18n: (
      /*gradio*/
      ctx[16].i18n
    ),
    line_breaks: (
      /*line_breaks*/
      ctx[14]
    ),
    column_widths: (
      /*column_widths*/
      ctx[15]
    ),
    upload: (
      /*func*/
      ctx[29]
    ),
    stream_handler: (
      /*func_1*/
      ctx[30]
    ),
    show_fullscreen_button: (
      /*show_fullscreen_button*/
      ctx[21]
    ),
    show_copy_button: (
      /*show_copy_button*/
      ctx[22]
    )
  };
  if (
    /*value_is_output*/
    ctx[1] !== void 0
  ) {
    table_props.value_is_output = /*value_is_output*/
    ctx[1];
  }
  table = new Table$1({ props: table_props });
  binding_callbacks.push(() => bind(table, "value_is_output", table_value_is_output_binding));
  table.$on(
    "change",
    /*change_handler*/
    ctx[32]
  );
  table.$on(
    "input",
    /*input_handler*/
    ctx[33]
  );
  table.$on(
    "select",
    /*select_handler*/
    ctx[34]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      create_component(table.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(table.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(table, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      589824 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        65536 && {
          autoscroll: (
            /*gradio*/
            ctx2[16].autoscroll
          )
        },
        dirty[0] & /*gradio*/
        65536 && { i18n: (
          /*gradio*/
          ctx2[16].i18n
        ) },
        dirty[0] & /*loading_status*/
        524288 && get_spread_object(
          /*loading_status*/
          ctx2[19]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const table_changes = {};
      if (dirty[0] & /*root*/
      8192)
        table_changes.root = /*root*/
        ctx2[13];
      if (dirty[0] & /*label*/
      128)
        table_changes.label = /*label*/
        ctx2[7];
      if (dirty[0] & /*show_label*/
      256)
        table_changes.show_label = /*show_label*/
        ctx2[8];
      if (dirty[0] & /*row_count*/
      64)
        table_changes.row_count = /*row_count*/
        ctx2[6];
      if (dirty[0] & /*col_count*/
      32)
        table_changes.col_count = /*col_count*/
        ctx2[5];
      if (dirty[0] & /*cell_values*/
      33554432)
        table_changes.values = /*cell_values*/
        ctx2[25];
      if (dirty[0] & /*display_value*/
      16777216)
        table_changes.display_value = /*display_value*/
        ctx2[24];
      if (dirty[0] & /*styling*/
      8388608)
        table_changes.styling = /*styling*/
        ctx2[23];
      if (dirty[0] & /*_headers*/
      67108864)
        table_changes.headers = /*_headers*/
        ctx2[26];
      if (dirty[0] & /*wrap*/
      512)
        table_changes.wrap = /*wrap*/
        ctx2[9];
      if (dirty[0] & /*datatype*/
      1024)
        table_changes.datatype = /*datatype*/
        ctx2[10];
      if (dirty[0] & /*latex_delimiters*/
      131072)
        table_changes.latex_delimiters = /*latex_delimiters*/
        ctx2[17];
      if (dirty[0] & /*interactive*/
      1048576)
        table_changes.editable = /*interactive*/
        ctx2[20];
      if (dirty[0] & /*max_height*/
      262144)
        table_changes.max_height = /*max_height*/
        ctx2[18];
      if (dirty[0] & /*gradio*/
      65536)
        table_changes.i18n = /*gradio*/
        ctx2[16].i18n;
      if (dirty[0] & /*line_breaks*/
      16384)
        table_changes.line_breaks = /*line_breaks*/
        ctx2[14];
      if (dirty[0] & /*column_widths*/
      32768)
        table_changes.column_widths = /*column_widths*/
        ctx2[15];
      if (dirty[0] & /*gradio*/
      65536)
        table_changes.upload = /*func*/
        ctx2[29];
      if (dirty[0] & /*gradio*/
      65536)
        table_changes.stream_handler = /*func_1*/
        ctx2[30];
      if (dirty[0] & /*show_fullscreen_button*/
      2097152)
        table_changes.show_fullscreen_button = /*show_fullscreen_button*/
        ctx2[21];
      if (dirty[0] & /*show_copy_button*/
      4194304)
        table_changes.show_copy_button = /*show_copy_button*/
        ctx2[22];
      if (!updating_value_is_output && dirty[0] & /*value_is_output*/
      2) {
        updating_value_is_output = true;
        table_changes.value_is_output = /*value_is_output*/
        ctx2[1];
        add_flush_callback(() => updating_value_is_output = false);
      }
      table.$set(table_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(table.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(table.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(statustracker, detaching);
      destroy_component(table, detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      visible: (
        /*visible*/
        ctx[4]
      ),
      padding: false,
      elem_id: (
        /*elem_id*/
        ctx[2]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[3]
      ),
      container: false,
      scale: (
        /*scale*/
        ctx[11]
      ),
      min_width: (
        /*min_width*/
        ctx[12]
      ),
      allow_overflow: false,
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(block.$$.fragment);
    },
    l(nodes) {
      claim_component(block.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(block, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const block_changes = {};
      if (dirty[0] & /*visible*/
      16)
        block_changes.visible = /*visible*/
        ctx2[4];
      if (dirty[0] & /*elem_id*/
      4)
        block_changes.elem_id = /*elem_id*/
        ctx2[2];
      if (dirty[0] & /*elem_classes*/
      8)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[3];
      if (dirty[0] & /*scale*/
      2048)
        block_changes.scale = /*scale*/
        ctx2[11];
      if (dirty[0] & /*min_width*/
      4096)
        block_changes.min_width = /*min_width*/
        ctx2[12];
      if (dirty[0] & /*root, label, show_label, row_count, col_count, cell_values, display_value, styling, _headers, wrap, datatype, latex_delimiters, interactive, max_height, gradio, line_breaks, column_widths, show_fullscreen_button, show_copy_button, value_is_output, value, loading_status*/
      134211555 | dirty[1] & /*$$scope*/
      16) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      block.$set(block_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(block.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(block.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(block, detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let _headers;
  let cell_values;
  let display_value;
  let styling;
  let { headers = [] } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = {
    data: [["", "", ""]],
    headers: ["1", "2", "3"],
    metadata: null
  } } = $$props;
  let { value_is_output = false } = $$props;
  let { col_count } = $$props;
  let { row_count } = $$props;
  let { label = null } = $$props;
  let { show_label = true } = $$props;
  let { wrap } = $$props;
  let { datatype } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { root } = $$props;
  let { line_breaks = true } = $$props;
  let { column_widths = [] } = $$props;
  let { gradio } = $$props;
  let { latex_delimiters } = $$props;
  let { max_height = void 0 } = $$props;
  let { loading_status } = $$props;
  let { interactive } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let { show_copy_button = false } = $$props;
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const func = (...args) => gradio.client.upload(...args);
  const func_1 = (...args) => gradio.client.stream(...args);
  function table_value_is_output_binding(value2) {
    value_is_output = value2;
    $$invalidate(1, value_is_output);
  }
  const change_handler = (e) => {
    $$invalidate(0, value = e.detail);
    gradio.dispatch("change");
  };
  const input_handler = (e) => gradio.dispatch("input");
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  $$self.$$set = ($$props2) => {
    if ("headers" in $$props2)
      $$invalidate(27, headers = $$props2.headers);
    if ("elem_id" in $$props2)
      $$invalidate(2, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(3, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(4, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("value_is_output" in $$props2)
      $$invalidate(1, value_is_output = $$props2.value_is_output);
    if ("col_count" in $$props2)
      $$invalidate(5, col_count = $$props2.col_count);
    if ("row_count" in $$props2)
      $$invalidate(6, row_count = $$props2.row_count);
    if ("label" in $$props2)
      $$invalidate(7, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(8, show_label = $$props2.show_label);
    if ("wrap" in $$props2)
      $$invalidate(9, wrap = $$props2.wrap);
    if ("datatype" in $$props2)
      $$invalidate(10, datatype = $$props2.datatype);
    if ("scale" in $$props2)
      $$invalidate(11, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(12, min_width = $$props2.min_width);
    if ("root" in $$props2)
      $$invalidate(13, root = $$props2.root);
    if ("line_breaks" in $$props2)
      $$invalidate(14, line_breaks = $$props2.line_breaks);
    if ("column_widths" in $$props2)
      $$invalidate(15, column_widths = $$props2.column_widths);
    if ("gradio" in $$props2)
      $$invalidate(16, gradio = $$props2.gradio);
    if ("latex_delimiters" in $$props2)
      $$invalidate(17, latex_delimiters = $$props2.latex_delimiters);
    if ("max_height" in $$props2)
      $$invalidate(18, max_height = $$props2.max_height);
    if ("loading_status" in $$props2)
      $$invalidate(19, loading_status = $$props2.loading_status);
    if ("interactive" in $$props2)
      $$invalidate(20, interactive = $$props2.interactive);
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(21, show_fullscreen_button = $$props2.show_fullscreen_button);
    if ("show_copy_button" in $$props2)
      $$invalidate(22, show_copy_button = $$props2.show_copy_button);
  };
  $$self.$$.update = () => {
    var _a, _b, _c, _d;
    if ($$self.$$.dirty[0] & /*value, headers*/
    134217729) {
      $$invalidate(26, _headers = [...value.headers || headers]);
    }
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      $$invalidate(25, cell_values = value.data ? [...value.data] : []);
    }
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      $$invalidate(24, display_value = ((_a = value == null ? void 0 : value.metadata) == null ? void 0 : _a.display_value) ? [...(_b = value == null ? void 0 : value.metadata) == null ? void 0 : _b.display_value] : null);
    }
    if ($$self.$$.dirty[0] & /*interactive, value*/
    1048577) {
      $$invalidate(23, styling = !interactive && ((_c = value == null ? void 0 : value.metadata) == null ? void 0 : _c.styling) ? [...(_d = value == null ? void 0 : value.metadata) == null ? void 0 : _d.styling] : null);
    }
  };
  return [
    value,
    value_is_output,
    elem_id,
    elem_classes,
    visible,
    col_count,
    row_count,
    label,
    show_label,
    wrap,
    datatype,
    scale,
    min_width,
    root,
    line_breaks,
    column_widths,
    gradio,
    latex_delimiters,
    max_height,
    loading_status,
    interactive,
    show_fullscreen_button,
    show_copy_button,
    styling,
    display_value,
    cell_values,
    _headers,
    headers,
    clear_status_handler,
    func,
    func_1,
    table_value_is_output_binding,
    change_handler,
    input_handler,
    select_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance,
      create_fragment,
      safe_not_equal,
      {
        headers: 27,
        elem_id: 2,
        elem_classes: 3,
        visible: 4,
        value: 0,
        value_is_output: 1,
        col_count: 5,
        row_count: 6,
        label: 7,
        show_label: 8,
        wrap: 9,
        datatype: 10,
        scale: 11,
        min_width: 12,
        root: 13,
        line_breaks: 14,
        column_widths: 15,
        gradio: 16,
        latex_delimiters: 17,
        max_height: 18,
        loading_status: 19,
        interactive: 20,
        show_fullscreen_button: 21,
        show_copy_button: 22
      },
      null,
      [-1, -1]
    );
  }
  get headers() {
    return this.$$.ctx[27];
  }
  set headers(headers) {
    this.$$set({ headers });
    flush();
  }
  get elem_id() {
    return this.$$.ctx[2];
  }
  set elem_id(elem_id) {
    this.$$set({ elem_id });
    flush();
  }
  get elem_classes() {
    return this.$$.ctx[3];
  }
  set elem_classes(elem_classes) {
    this.$$set({ elem_classes });
    flush();
  }
  get visible() {
    return this.$$.ctx[4];
  }
  set visible(visible) {
    this.$$set({ visible });
    flush();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(value) {
    this.$$set({ value });
    flush();
  }
  get value_is_output() {
    return this.$$.ctx[1];
  }
  set value_is_output(value_is_output) {
    this.$$set({ value_is_output });
    flush();
  }
  get col_count() {
    return this.$$.ctx[5];
  }
  set col_count(col_count) {
    this.$$set({ col_count });
    flush();
  }
  get row_count() {
    return this.$$.ctx[6];
  }
  set row_count(row_count) {
    this.$$set({ row_count });
    flush();
  }
  get label() {
    return this.$$.ctx[7];
  }
  set label(label) {
    this.$$set({ label });
    flush();
  }
  get show_label() {
    return this.$$.ctx[8];
  }
  set show_label(show_label) {
    this.$$set({ show_label });
    flush();
  }
  get wrap() {
    return this.$$.ctx[9];
  }
  set wrap(wrap) {
    this.$$set({ wrap });
    flush();
  }
  get datatype() {
    return this.$$.ctx[10];
  }
  set datatype(datatype) {
    this.$$set({ datatype });
    flush();
  }
  get scale() {
    return this.$$.ctx[11];
  }
  set scale(scale) {
    this.$$set({ scale });
    flush();
  }
  get min_width() {
    return this.$$.ctx[12];
  }
  set min_width(min_width) {
    this.$$set({ min_width });
    flush();
  }
  get root() {
    return this.$$.ctx[13];
  }
  set root(root) {
    this.$$set({ root });
    flush();
  }
  get line_breaks() {
    return this.$$.ctx[14];
  }
  set line_breaks(line_breaks) {
    this.$$set({ line_breaks });
    flush();
  }
  get column_widths() {
    return this.$$.ctx[15];
  }
  set column_widths(column_widths) {
    this.$$set({ column_widths });
    flush();
  }
  get gradio() {
    return this.$$.ctx[16];
  }
  set gradio(gradio) {
    this.$$set({ gradio });
    flush();
  }
  get latex_delimiters() {
    return this.$$.ctx[17];
  }
  set latex_delimiters(latex_delimiters) {
    this.$$set({ latex_delimiters });
    flush();
  }
  get max_height() {
    return this.$$.ctx[18];
  }
  set max_height(max_height) {
    this.$$set({ max_height });
    flush();
  }
  get loading_status() {
    return this.$$.ctx[19];
  }
  set loading_status(loading_status) {
    this.$$set({ loading_status });
    flush();
  }
  get interactive() {
    return this.$$.ctx[20];
  }
  set interactive(interactive) {
    this.$$set({ interactive });
    flush();
  }
  get show_fullscreen_button() {
    return this.$$.ctx[21];
  }
  set show_fullscreen_button(show_fullscreen_button) {
    this.$$set({ show_fullscreen_button });
    flush();
  }
  get show_copy_button() {
    return this.$$.ctx[22];
  }
  set show_copy_button(show_copy_button) {
    this.$$set({ show_copy_button });
    flush();
  }
}
export {
  Table$1 as BaseDataFrame,
  default2 as BaseExample,
  Index as default
};
