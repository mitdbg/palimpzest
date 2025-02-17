import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, ensure_array_like, element, create_component, space, claim_element, claim_component, claim_space, toggle_class, mount_component, set_input_value, listen, transition_in, group_outros, check_outros, transition_out, destroy_component, destroy_each, run_all, createEventDispatcher, afterUpdate, text, claim_text, set_data, binding_callbacks, prevent_default, assign, empty, get_spread_update, get_spread_object, bind, add_flush_callback } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { i as BlockTitle, D as DropdownOptions, q as handle_filter, r as handle_change, t as DropdownArrow, v as handle_shared_keys, B as Block, S as Static, w as Dropdown } from "./2.8WKXZUMv.js";
import { default as default2 } from "./Example.DxNyyiKy.js";
function create_fragment$2(ctx) {
  let svg;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        xmlns: true,
        viewBox: true,
        width: true,
        height: true
      });
      var svg_nodes = children(svg);
      path = claim_svg_element(svg_nodes, "path", { d: true });
      children(path).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(path, "d", "M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "width", "100%");
      attr(svg, "height", "100%");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, path);
    },
    p: noop,
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
class Remove extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$2, safe_not_equal, {});
  }
}
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[41] = list[i];
  return child_ctx;
}
function create_default_slot$1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[0]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[0]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      1)
        set_data(
          t,
          /*label*/
          ctx2[0]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_else_block$1(ctx) {
  let t_value = (
    /*s*/
    ctx[41] + ""
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
      if (dirty[0] & /*selected_indices*/
      8192 && t_value !== (t_value = /*s*/
      ctx2[41] + ""))
        set_data(t, t_value);
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_3(ctx) {
  let t_value = (
    /*choices_names*/
    ctx[16][
      /*s*/
      ctx[41]
    ] + ""
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
      if (dirty[0] & /*choices_names, selected_indices*/
      73728 && t_value !== (t_value = /*choices_names*/
      ctx2[16][
        /*s*/
        ctx2[41]
      ] + ""))
        set_data(t, t_value);
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_2(ctx) {
  let div;
  let remove;
  let div_title_value;
  let current;
  let mounted;
  let dispose;
  remove = new Remove({});
  function click_handler() {
    return (
      /*click_handler*/
      ctx[32](
        /*s*/
        ctx[41]
      )
    );
  }
  function keydown_handler(...args) {
    return (
      /*keydown_handler*/
      ctx[33](
        /*s*/
        ctx[41],
        ...args
      )
    );
  }
  return {
    c() {
      div = element("div");
      create_component(remove.$$.fragment);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {
        class: true,
        role: true,
        tabindex: true,
        title: true
      });
      var div_nodes = children(div);
      claim_component(remove.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "token-remove svelte-1scun43");
      attr(div, "role", "button");
      attr(div, "tabindex", "0");
      attr(div, "title", div_title_value = /*i18n*/
      ctx[9]("common.remove") + " " + /*s*/
      ctx[41]);
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      mount_component(remove, div, null);
      current = true;
      if (!mounted) {
        dispose = [
          listen(div, "click", prevent_default(click_handler)),
          listen(div, "keydown", prevent_default(keydown_handler))
        ];
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (!current || dirty[0] & /*i18n, selected_indices*/
      8704 && div_title_value !== (div_title_value = /*i18n*/
      ctx[9]("common.remove") + " " + /*s*/
      ctx[41])) {
        attr(div, "title", div_title_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(remove.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(remove.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_component(remove);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_each_block(ctx) {
  let div;
  let span;
  let t;
  let current;
  function select_block_type(ctx2, dirty) {
    if (typeof /*s*/
    ctx2[41] === "number")
      return create_if_block_3;
    return create_else_block$1;
  }
  let current_block_type = select_block_type(ctx);
  let if_block0 = current_block_type(ctx);
  let if_block1 = !/*disabled*/
  ctx[4] && create_if_block_2(ctx);
  return {
    c() {
      div = element("div");
      span = element("span");
      if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      span = claim_element(div_nodes, "SPAN", { class: true });
      var span_nodes = children(span);
      if_block0.l(span_nodes);
      span_nodes.forEach(detach);
      t = claim_space(div_nodes);
      if (if_block1)
        if_block1.l(div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(span, "class", "svelte-1scun43");
      attr(div, "class", "token svelte-1scun43");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      append_hydration(div, span);
      if_block0.m(span, null);
      append_hydration(div, t);
      if (if_block1)
        if_block1.m(div, null);
      current = true;
    },
    p(ctx2, dirty) {
      if (current_block_type === (current_block_type = select_block_type(ctx2)) && if_block0) {
        if_block0.p(ctx2, dirty);
      } else {
        if_block0.d(1);
        if_block0 = current_block_type(ctx2);
        if (if_block0) {
          if_block0.c();
          if_block0.m(span, null);
        }
      }
      if (!/*disabled*/
      ctx2[4]) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*disabled*/
          16) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_2(ctx2);
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
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      if_block0.d();
      if (if_block1)
        if_block1.d();
    }
  };
}
function create_if_block$1(ctx) {
  let t;
  let span;
  let dropdownarrow;
  let current;
  let if_block = (
    /*selected_indices*/
    ctx[13].length > 0 && create_if_block_1(ctx)
  );
  dropdownarrow = new DropdownArrow({});
  return {
    c() {
      if (if_block)
        if_block.c();
      t = space();
      span = element("span");
      create_component(dropdownarrow.$$.fragment);
      this.h();
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      t = claim_space(nodes);
      span = claim_element(nodes, "SPAN", { class: true });
      var span_nodes = children(span);
      claim_component(dropdownarrow.$$.fragment, span_nodes);
      span_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(span, "class", "icon-wrap svelte-1scun43");
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t, anchor);
      insert_hydration(target, span, anchor);
      mount_component(dropdownarrow, span, null);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*selected_indices*/
        ctx2[13].length > 0
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty[0] & /*selected_indices*/
          8192) {
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
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(dropdownarrow.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(dropdownarrow.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(span);
      }
      if (if_block)
        if_block.d(detaching);
      destroy_component(dropdownarrow);
    }
  };
}
function create_if_block_1(ctx) {
  let div;
  let remove;
  let div_title_value;
  let current;
  let mounted;
  let dispose;
  remove = new Remove({});
  return {
    c() {
      div = element("div");
      create_component(remove.$$.fragment);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {
        role: true,
        tabindex: true,
        class: true,
        title: true
      });
      var div_nodes = children(div);
      claim_component(remove.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "role", "button");
      attr(div, "tabindex", "0");
      attr(div, "class", "token-remove remove-all svelte-1scun43");
      attr(div, "title", div_title_value = /*i18n*/
      ctx[9]("common.clear"));
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      mount_component(remove, div, null);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            div,
            "click",
            /*remove_all*/
            ctx[22]
          ),
          listen(
            div,
            "keydown",
            /*keydown_handler_1*/
            ctx[37]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (!current || dirty[0] & /*i18n*/
      512 && div_title_value !== (div_title_value = /*i18n*/
      ctx2[9]("common.clear"))) {
        attr(div, "title", div_title_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(remove.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(remove.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_component(remove);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_fragment$1(ctx) {
  let label_1;
  let blocktitle;
  let t0;
  let div2;
  let div1;
  let t1;
  let div0;
  let input;
  let input_readonly_value;
  let t2;
  let t3;
  let dropdownoptions;
  let current;
  let mounted;
  let dispose;
  blocktitle = new BlockTitle({
    props: {
      root: (
        /*root*/
        ctx[10]
      ),
      show_label: (
        /*show_label*/
        ctx[5]
      ),
      info: (
        /*info*/
        ctx[1]
      ),
      $$slots: { default: [create_default_slot$1] },
      $$scope: { ctx }
    }
  });
  let each_value = ensure_array_like(
    /*selected_indices*/
    ctx[13]
  );
  let each_blocks = [];
  for (let i = 0; i < each_value.length; i += 1) {
    each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
  }
  const out = (i) => transition_out(each_blocks[i], 1, 1, () => {
    each_blocks[i] = null;
  });
  let if_block = !/*disabled*/
  ctx[4] && create_if_block$1(ctx);
  dropdownoptions = new DropdownOptions({
    props: {
      show_options: (
        /*show_options*/
        ctx[15]
      ),
      choices: (
        /*choices*/
        ctx[3]
      ),
      filtered_indices: (
        /*filtered_indices*/
        ctx[12]
      ),
      disabled: (
        /*disabled*/
        ctx[4]
      ),
      selected_indices: (
        /*selected_indices*/
        ctx[13]
      ),
      active_index: (
        /*active_index*/
        ctx[17]
      )
    }
  });
  dropdownoptions.$on(
    "change",
    /*handle_option_selected*/
    ctx[21]
  );
  return {
    c() {
      label_1 = element("label");
      create_component(blocktitle.$$.fragment);
      t0 = space();
      div2 = element("div");
      div1 = element("div");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      t1 = space();
      div0 = element("div");
      input = element("input");
      t2 = space();
      if (if_block)
        if_block.c();
      t3 = space();
      create_component(dropdownoptions.$$.fragment);
      this.h();
    },
    l(nodes) {
      label_1 = claim_element(nodes, "LABEL", { class: true });
      var label_1_nodes = children(label_1);
      claim_component(blocktitle.$$.fragment, label_1_nodes);
      t0 = claim_space(label_1_nodes);
      div2 = claim_element(label_1_nodes, "DIV", { class: true });
      var div2_nodes = children(div2);
      div1 = claim_element(div2_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(div1_nodes);
      }
      t1 = claim_space(div1_nodes);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      input = claim_element(div0_nodes, "INPUT", { class: true, autocomplete: true });
      t2 = claim_space(div0_nodes);
      if (if_block)
        if_block.l(div0_nodes);
      div0_nodes.forEach(detach);
      div1_nodes.forEach(detach);
      t3 = claim_space(div2_nodes);
      claim_component(dropdownoptions.$$.fragment, div2_nodes);
      div2_nodes.forEach(detach);
      label_1_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(input, "class", "border-none svelte-1scun43");
      input.disabled = /*disabled*/
      ctx[4];
      attr(input, "autocomplete", "off");
      input.readOnly = input_readonly_value = !/*filterable*/
      ctx[8];
      toggle_class(input, "subdued", !/*choices_names*/
      ctx[16].includes(
        /*input_text*/
        ctx[11]
      ) && !/*allow_custom_value*/
      ctx[7] || /*selected_indices*/
      ctx[13].length === /*max_choices*/
      ctx[2]);
      attr(div0, "class", "secondary-wrap svelte-1scun43");
      attr(div1, "class", "wrap-inner svelte-1scun43");
      toggle_class(
        div1,
        "show_options",
        /*show_options*/
        ctx[15]
      );
      attr(div2, "class", "wrap svelte-1scun43");
      attr(label_1, "class", "svelte-1scun43");
      toggle_class(
        label_1,
        "container",
        /*container*/
        ctx[6]
      );
    },
    m(target, anchor) {
      insert_hydration(target, label_1, anchor);
      mount_component(blocktitle, label_1, null);
      append_hydration(label_1, t0);
      append_hydration(label_1, div2);
      append_hydration(div2, div1);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(div1, null);
        }
      }
      append_hydration(div1, t1);
      append_hydration(div1, div0);
      append_hydration(div0, input);
      set_input_value(
        input,
        /*input_text*/
        ctx[11]
      );
      ctx[35](input);
      append_hydration(div0, t2);
      if (if_block)
        if_block.m(div0, null);
      append_hydration(div2, t3);
      mount_component(dropdownoptions, div2, null);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler*/
            ctx[34]
          ),
          listen(
            input,
            "keydown",
            /*handle_key_down*/
            ctx[24]
          ),
          listen(
            input,
            "keyup",
            /*keyup_handler*/
            ctx[36]
          ),
          listen(
            input,
            "blur",
            /*handle_blur*/
            ctx[19]
          ),
          listen(
            input,
            "focus",
            /*handle_focus*/
            ctx[23]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      const blocktitle_changes = {};
      if (dirty[0] & /*root*/
      1024)
        blocktitle_changes.root = /*root*/
        ctx2[10];
      if (dirty[0] & /*show_label*/
      32)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[5];
      if (dirty[0] & /*info*/
      2)
        blocktitle_changes.info = /*info*/
        ctx2[1];
      if (dirty[0] & /*label*/
      1 | dirty[1] & /*$$scope*/
      8192) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      if (dirty[0] & /*i18n, selected_indices, remove_selected_choice, disabled, choices_names*/
      1122832) {
        each_value = ensure_array_like(
          /*selected_indices*/
          ctx2[13]
        );
        let i;
        for (i = 0; i < each_value.length; i += 1) {
          const child_ctx = get_each_context(ctx2, each_value, i);
          if (each_blocks[i]) {
            each_blocks[i].p(child_ctx, dirty);
            transition_in(each_blocks[i], 1);
          } else {
            each_blocks[i] = create_each_block(child_ctx);
            each_blocks[i].c();
            transition_in(each_blocks[i], 1);
            each_blocks[i].m(div1, t1);
          }
        }
        group_outros();
        for (i = each_value.length; i < each_blocks.length; i += 1) {
          out(i);
        }
        check_outros();
      }
      if (!current || dirty[0] & /*disabled*/
      16) {
        input.disabled = /*disabled*/
        ctx2[4];
      }
      if (!current || dirty[0] & /*filterable*/
      256 && input_readonly_value !== (input_readonly_value = !/*filterable*/
      ctx2[8])) {
        input.readOnly = input_readonly_value;
      }
      if (dirty[0] & /*input_text*/
      2048 && input.value !== /*input_text*/
      ctx2[11]) {
        set_input_value(
          input,
          /*input_text*/
          ctx2[11]
        );
      }
      if (!current || dirty[0] & /*choices_names, input_text, allow_custom_value, selected_indices, max_choices*/
      75908) {
        toggle_class(input, "subdued", !/*choices_names*/
        ctx2[16].includes(
          /*input_text*/
          ctx2[11]
        ) && !/*allow_custom_value*/
        ctx2[7] || /*selected_indices*/
        ctx2[13].length === /*max_choices*/
        ctx2[2]);
      }
      if (!/*disabled*/
      ctx2[4]) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty[0] & /*disabled*/
          16) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block$1(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(div0, null);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (!current || dirty[0] & /*show_options*/
      32768) {
        toggle_class(
          div1,
          "show_options",
          /*show_options*/
          ctx2[15]
        );
      }
      const dropdownoptions_changes = {};
      if (dirty[0] & /*show_options*/
      32768)
        dropdownoptions_changes.show_options = /*show_options*/
        ctx2[15];
      if (dirty[0] & /*choices*/
      8)
        dropdownoptions_changes.choices = /*choices*/
        ctx2[3];
      if (dirty[0] & /*filtered_indices*/
      4096)
        dropdownoptions_changes.filtered_indices = /*filtered_indices*/
        ctx2[12];
      if (dirty[0] & /*disabled*/
      16)
        dropdownoptions_changes.disabled = /*disabled*/
        ctx2[4];
      if (dirty[0] & /*selected_indices*/
      8192)
        dropdownoptions_changes.selected_indices = /*selected_indices*/
        ctx2[13];
      if (dirty[0] & /*active_index*/
      131072)
        dropdownoptions_changes.active_index = /*active_index*/
        ctx2[17];
      dropdownoptions.$set(dropdownoptions_changes);
      if (!current || dirty[0] & /*container*/
      64) {
        toggle_class(
          label_1,
          "container",
          /*container*/
          ctx2[6]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(blocktitle.$$.fragment, local);
      for (let i = 0; i < each_value.length; i += 1) {
        transition_in(each_blocks[i]);
      }
      transition_in(if_block);
      transition_in(dropdownoptions.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(blocktitle.$$.fragment, local);
      each_blocks = each_blocks.filter(Boolean);
      for (let i = 0; i < each_blocks.length; i += 1) {
        transition_out(each_blocks[i]);
      }
      transition_out(if_block);
      transition_out(dropdownoptions.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(label_1);
      }
      destroy_component(blocktitle);
      destroy_each(each_blocks, detaching);
      ctx[35](null);
      if (if_block)
        if_block.d();
      destroy_component(dropdownoptions);
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let { label } = $$props;
  let { info = void 0 } = $$props;
  let { value = [] } = $$props;
  let old_value = [];
  let { value_is_output = false } = $$props;
  let { max_choices = null } = $$props;
  let { choices } = $$props;
  let old_choices;
  let { disabled = false } = $$props;
  let { show_label } = $$props;
  let { container = true } = $$props;
  let { allow_custom_value = false } = $$props;
  let { filterable = true } = $$props;
  let { i18n } = $$props;
  let { root } = $$props;
  let filter_input;
  let input_text = "";
  let old_input_text = "";
  let show_options = false;
  let choices_names;
  let choices_values;
  let filtered_indices = [];
  let active_index = null;
  let selected_indices = [];
  let old_selected_index = [];
  const dispatch = createEventDispatcher();
  if (Array.isArray(value)) {
    value.forEach((element2) => {
      const index = choices.map((c) => c[1]).indexOf(element2);
      if (index !== -1) {
        selected_indices.push(index);
      } else {
        selected_indices.push(element2);
      }
    });
  }
  function handle_blur() {
    if (!allow_custom_value) {
      $$invalidate(11, input_text = "");
    }
    if (allow_custom_value && input_text !== "") {
      add_selected_choice(input_text);
      $$invalidate(11, input_text = "");
    }
    $$invalidate(15, show_options = false);
    $$invalidate(17, active_index = null);
    dispatch("blur");
  }
  function remove_selected_choice(option_index) {
    $$invalidate(13, selected_indices = selected_indices.filter((v) => v !== option_index));
    dispatch("select", {
      index: typeof option_index === "number" ? option_index : -1,
      value: typeof option_index === "number" ? choices_values[option_index] : option_index,
      selected: false
    });
  }
  function add_selected_choice(option_index) {
    if (max_choices === null || selected_indices.length < max_choices) {
      $$invalidate(13, selected_indices = [...selected_indices, option_index]);
      dispatch("select", {
        index: typeof option_index === "number" ? option_index : -1,
        value: typeof option_index === "number" ? choices_values[option_index] : option_index,
        selected: true
      });
    }
    if (selected_indices.length === max_choices) {
      $$invalidate(15, show_options = false);
      $$invalidate(17, active_index = null);
      filter_input.blur();
    }
  }
  function handle_option_selected(e) {
    const option_index = parseInt(e.detail.target.dataset.index);
    add_or_remove_index(option_index);
  }
  function add_or_remove_index(option_index) {
    if (selected_indices.includes(option_index)) {
      remove_selected_choice(option_index);
    } else {
      add_selected_choice(option_index);
    }
    $$invalidate(11, input_text = "");
  }
  function remove_all(e) {
    $$invalidate(13, selected_indices = []);
    $$invalidate(11, input_text = "");
    e.preventDefault();
  }
  function handle_focus(e) {
    $$invalidate(12, filtered_indices = choices.map((_2, i) => i));
    if (max_choices === null || selected_indices.length < max_choices) {
      $$invalidate(15, show_options = true);
    }
    dispatch("focus");
  }
  function handle_key_down(e) {
    $$invalidate(15, [show_options, active_index] = handle_shared_keys(e, active_index, filtered_indices), show_options, ($$invalidate(17, active_index), $$invalidate(3, choices), $$invalidate(28, old_choices), $$invalidate(11, input_text), $$invalidate(29, old_input_text), $$invalidate(7, allow_custom_value), $$invalidate(12, filtered_indices)));
    if (e.key === "Enter") {
      if (active_index !== null) {
        add_or_remove_index(active_index);
      } else {
        if (allow_custom_value) {
          add_selected_choice(input_text);
          $$invalidate(11, input_text = "");
        }
      }
    }
    if (e.key === "Backspace" && input_text === "") {
      $$invalidate(13, selected_indices = [...selected_indices.slice(0, -1)]);
    }
    if (selected_indices.length === max_choices) {
      $$invalidate(15, show_options = false);
      $$invalidate(17, active_index = null);
    }
  }
  function set_selected_indices() {
    if (value === void 0) {
      $$invalidate(13, selected_indices = []);
    } else if (Array.isArray(value)) {
      $$invalidate(13, selected_indices = value.map((v) => {
        const index = choices_values.indexOf(v);
        if (index !== -1) {
          return index;
        }
        if (allow_custom_value) {
          return v;
        }
        return void 0;
      }).filter((val) => val !== void 0));
    }
  }
  afterUpdate(() => {
    $$invalidate(26, value_is_output = false);
  });
  const click_handler = (s) => remove_selected_choice(s);
  const keydown_handler = (s, event) => {
    if (event.key === "Enter") {
      remove_selected_choice(s);
    }
  };
  function input_input_handler() {
    input_text = this.value;
    $$invalidate(11, input_text);
  }
  function input_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      filter_input = $$value;
      $$invalidate(14, filter_input);
    });
  }
  const keyup_handler = (e) => dispatch("key_up", { key: e.key, input_value: input_text });
  const keydown_handler_1 = (event) => {
    if (event.key === "Enter") {
      remove_all(event);
    }
  };
  $$self.$$set = ($$props2) => {
    if ("label" in $$props2)
      $$invalidate(0, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(1, info = $$props2.info);
    if ("value" in $$props2)
      $$invalidate(25, value = $$props2.value);
    if ("value_is_output" in $$props2)
      $$invalidate(26, value_is_output = $$props2.value_is_output);
    if ("max_choices" in $$props2)
      $$invalidate(2, max_choices = $$props2.max_choices);
    if ("choices" in $$props2)
      $$invalidate(3, choices = $$props2.choices);
    if ("disabled" in $$props2)
      $$invalidate(4, disabled = $$props2.disabled);
    if ("show_label" in $$props2)
      $$invalidate(5, show_label = $$props2.show_label);
    if ("container" in $$props2)
      $$invalidate(6, container = $$props2.container);
    if ("allow_custom_value" in $$props2)
      $$invalidate(7, allow_custom_value = $$props2.allow_custom_value);
    if ("filterable" in $$props2)
      $$invalidate(8, filterable = $$props2.filterable);
    if ("i18n" in $$props2)
      $$invalidate(9, i18n = $$props2.i18n);
    if ("root" in $$props2)
      $$invalidate(10, root = $$props2.root);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*choices*/
    8) {
      {
        $$invalidate(16, choices_names = choices.map((c) => c[0]));
        $$invalidate(30, choices_values = choices.map((c) => c[1]));
      }
    }
    if ($$self.$$.dirty[0] & /*choices, old_choices, input_text, old_input_text, allow_custom_value, filtered_indices*/
    805312648) {
      {
        if (choices !== old_choices || input_text !== old_input_text) {
          $$invalidate(12, filtered_indices = handle_filter(choices, input_text));
          $$invalidate(28, old_choices = choices);
          $$invalidate(29, old_input_text = input_text);
          if (!allow_custom_value) {
            $$invalidate(17, active_index = filtered_indices[0]);
          }
        }
      }
    }
    if ($$self.$$.dirty[0] & /*selected_indices, choices_values*/
    1073750016 | $$self.$$.dirty[1] & /*old_selected_index*/
    1) {
      {
        if (JSON.stringify(selected_indices) != JSON.stringify(old_selected_index)) {
          $$invalidate(25, value = selected_indices.map((index) => typeof index === "number" ? choices_values[index] : index));
          $$invalidate(31, old_selected_index = selected_indices.slice());
        }
      }
    }
    if ($$self.$$.dirty[0] & /*value, old_value, value_is_output*/
    234881024) {
      {
        if (JSON.stringify(value) != JSON.stringify(old_value)) {
          handle_change(dispatch, value, value_is_output);
          $$invalidate(27, old_value = Array.isArray(value) ? value.slice() : value);
        }
      }
    }
    if ($$self.$$.dirty[0] & /*value*/
    33554432) {
      set_selected_indices();
    }
  };
  return [
    label,
    info,
    max_choices,
    choices,
    disabled,
    show_label,
    container,
    allow_custom_value,
    filterable,
    i18n,
    root,
    input_text,
    filtered_indices,
    selected_indices,
    filter_input,
    show_options,
    choices_names,
    active_index,
    dispatch,
    handle_blur,
    remove_selected_choice,
    handle_option_selected,
    remove_all,
    handle_focus,
    handle_key_down,
    value,
    value_is_output,
    old_value,
    old_choices,
    old_input_text,
    choices_values,
    old_selected_index,
    click_handler,
    keydown_handler,
    input_input_handler,
    input_binding,
    keyup_handler,
    keydown_handler_1
  ];
}
class Multiselect extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance$1,
      create_fragment$1,
      safe_not_equal,
      {
        label: 0,
        info: 1,
        value: 25,
        value_is_output: 26,
        max_choices: 2,
        choices: 3,
        disabled: 4,
        show_label: 5,
        container: 6,
        allow_custom_value: 7,
        filterable: 8,
        i18n: 9,
        root: 10
      },
      null,
      [-1, -1]
    );
  }
}
function create_else_block(ctx) {
  let dropdown;
  let updating_value;
  let updating_value_is_output;
  let current;
  function dropdown_value_binding(value) {
    ctx[29](value);
  }
  function dropdown_value_is_output_binding(value) {
    ctx[30](value);
  }
  let dropdown_props = {
    choices: (
      /*choices*/
      ctx[9]
    ),
    label: (
      /*label*/
      ctx[2]
    ),
    root: (
      /*root*/
      ctx[17]
    ),
    info: (
      /*info*/
      ctx[3]
    ),
    show_label: (
      /*show_label*/
      ctx[10]
    ),
    filterable: (
      /*filterable*/
      ctx[11]
    ),
    allow_custom_value: (
      /*allow_custom_value*/
      ctx[16]
    ),
    container: (
      /*container*/
      ctx[12]
    ),
    disabled: !/*interactive*/
    ctx[19]
  };
  if (
    /*value*/
    ctx[0] !== void 0
  ) {
    dropdown_props.value = /*value*/
    ctx[0];
  }
  if (
    /*value_is_output*/
    ctx[1] !== void 0
  ) {
    dropdown_props.value_is_output = /*value_is_output*/
    ctx[1];
  }
  dropdown = new Dropdown({ props: dropdown_props });
  binding_callbacks.push(() => bind(dropdown, "value", dropdown_value_binding));
  binding_callbacks.push(() => bind(dropdown, "value_is_output", dropdown_value_is_output_binding));
  dropdown.$on(
    "change",
    /*change_handler_1*/
    ctx[31]
  );
  dropdown.$on(
    "input",
    /*input_handler_1*/
    ctx[32]
  );
  dropdown.$on(
    "select",
    /*select_handler_1*/
    ctx[33]
  );
  dropdown.$on(
    "blur",
    /*blur_handler_1*/
    ctx[34]
  );
  dropdown.$on(
    "focus",
    /*focus_handler_1*/
    ctx[35]
  );
  dropdown.$on(
    "key_up",
    /*key_up_handler_1*/
    ctx[36]
  );
  return {
    c() {
      create_component(dropdown.$$.fragment);
    },
    l(nodes) {
      claim_component(dropdown.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(dropdown, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const dropdown_changes = {};
      if (dirty[0] & /*choices*/
      512)
        dropdown_changes.choices = /*choices*/
        ctx2[9];
      if (dirty[0] & /*label*/
      4)
        dropdown_changes.label = /*label*/
        ctx2[2];
      if (dirty[0] & /*root*/
      131072)
        dropdown_changes.root = /*root*/
        ctx2[17];
      if (dirty[0] & /*info*/
      8)
        dropdown_changes.info = /*info*/
        ctx2[3];
      if (dirty[0] & /*show_label*/
      1024)
        dropdown_changes.show_label = /*show_label*/
        ctx2[10];
      if (dirty[0] & /*filterable*/
      2048)
        dropdown_changes.filterable = /*filterable*/
        ctx2[11];
      if (dirty[0] & /*allow_custom_value*/
      65536)
        dropdown_changes.allow_custom_value = /*allow_custom_value*/
        ctx2[16];
      if (dirty[0] & /*container*/
      4096)
        dropdown_changes.container = /*container*/
        ctx2[12];
      if (dirty[0] & /*interactive*/
      524288)
        dropdown_changes.disabled = !/*interactive*/
        ctx2[19];
      if (!updating_value && dirty[0] & /*value*/
      1) {
        updating_value = true;
        dropdown_changes.value = /*value*/
        ctx2[0];
        add_flush_callback(() => updating_value = false);
      }
      if (!updating_value_is_output && dirty[0] & /*value_is_output*/
      2) {
        updating_value_is_output = true;
        dropdown_changes.value_is_output = /*value_is_output*/
        ctx2[1];
        add_flush_callback(() => updating_value_is_output = false);
      }
      dropdown.$set(dropdown_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(dropdown.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(dropdown.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(dropdown, detaching);
    }
  };
}
function create_if_block(ctx) {
  let multiselect_1;
  let updating_value;
  let updating_value_is_output;
  let current;
  function multiselect_1_value_binding(value) {
    ctx[21](value);
  }
  function multiselect_1_value_is_output_binding(value) {
    ctx[22](value);
  }
  let multiselect_1_props = {
    choices: (
      /*choices*/
      ctx[9]
    ),
    max_choices: (
      /*max_choices*/
      ctx[8]
    ),
    root: (
      /*root*/
      ctx[17]
    ),
    label: (
      /*label*/
      ctx[2]
    ),
    info: (
      /*info*/
      ctx[3]
    ),
    show_label: (
      /*show_label*/
      ctx[10]
    ),
    allow_custom_value: (
      /*allow_custom_value*/
      ctx[16]
    ),
    filterable: (
      /*filterable*/
      ctx[11]
    ),
    container: (
      /*container*/
      ctx[12]
    ),
    i18n: (
      /*gradio*/
      ctx[18].i18n
    ),
    disabled: !/*interactive*/
    ctx[19]
  };
  if (
    /*value*/
    ctx[0] !== void 0
  ) {
    multiselect_1_props.value = /*value*/
    ctx[0];
  }
  if (
    /*value_is_output*/
    ctx[1] !== void 0
  ) {
    multiselect_1_props.value_is_output = /*value_is_output*/
    ctx[1];
  }
  multiselect_1 = new Multiselect({ props: multiselect_1_props });
  binding_callbacks.push(() => bind(multiselect_1, "value", multiselect_1_value_binding));
  binding_callbacks.push(() => bind(multiselect_1, "value_is_output", multiselect_1_value_is_output_binding));
  multiselect_1.$on(
    "change",
    /*change_handler*/
    ctx[23]
  );
  multiselect_1.$on(
    "input",
    /*input_handler*/
    ctx[24]
  );
  multiselect_1.$on(
    "select",
    /*select_handler*/
    ctx[25]
  );
  multiselect_1.$on(
    "blur",
    /*blur_handler*/
    ctx[26]
  );
  multiselect_1.$on(
    "focus",
    /*focus_handler*/
    ctx[27]
  );
  multiselect_1.$on(
    "key_up",
    /*key_up_handler*/
    ctx[28]
  );
  return {
    c() {
      create_component(multiselect_1.$$.fragment);
    },
    l(nodes) {
      claim_component(multiselect_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(multiselect_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const multiselect_1_changes = {};
      if (dirty[0] & /*choices*/
      512)
        multiselect_1_changes.choices = /*choices*/
        ctx2[9];
      if (dirty[0] & /*max_choices*/
      256)
        multiselect_1_changes.max_choices = /*max_choices*/
        ctx2[8];
      if (dirty[0] & /*root*/
      131072)
        multiselect_1_changes.root = /*root*/
        ctx2[17];
      if (dirty[0] & /*label*/
      4)
        multiselect_1_changes.label = /*label*/
        ctx2[2];
      if (dirty[0] & /*info*/
      8)
        multiselect_1_changes.info = /*info*/
        ctx2[3];
      if (dirty[0] & /*show_label*/
      1024)
        multiselect_1_changes.show_label = /*show_label*/
        ctx2[10];
      if (dirty[0] & /*allow_custom_value*/
      65536)
        multiselect_1_changes.allow_custom_value = /*allow_custom_value*/
        ctx2[16];
      if (dirty[0] & /*filterable*/
      2048)
        multiselect_1_changes.filterable = /*filterable*/
        ctx2[11];
      if (dirty[0] & /*container*/
      4096)
        multiselect_1_changes.container = /*container*/
        ctx2[12];
      if (dirty[0] & /*gradio*/
      262144)
        multiselect_1_changes.i18n = /*gradio*/
        ctx2[18].i18n;
      if (dirty[0] & /*interactive*/
      524288)
        multiselect_1_changes.disabled = !/*interactive*/
        ctx2[19];
      if (!updating_value && dirty[0] & /*value*/
      1) {
        updating_value = true;
        multiselect_1_changes.value = /*value*/
        ctx2[0];
        add_flush_callback(() => updating_value = false);
      }
      if (!updating_value_is_output && dirty[0] & /*value_is_output*/
      2) {
        updating_value_is_output = true;
        multiselect_1_changes.value_is_output = /*value_is_output*/
        ctx2[1];
        add_flush_callback(() => updating_value_is_output = false);
      }
      multiselect_1.$set(multiselect_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(multiselect_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(multiselect_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(multiselect_1, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let statustracker;
  let t;
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[18].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[18].i18n
    ) },
    /*loading_status*/
    ctx[15]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[20]
  );
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*multiselect*/
      ctx2[7]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      294912 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        262144 && {
          autoscroll: (
            /*gradio*/
            ctx2[18].autoscroll
          )
        },
        dirty[0] & /*gradio*/
        262144 && { i18n: (
          /*gradio*/
          ctx2[18].i18n
        ) },
        dirty[0] & /*loading_status*/
        32768 && get_spread_object(
          /*loading_status*/
          ctx2[15]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
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
      transition_in(statustracker.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block_anchor);
      }
      destroy_component(statustracker, detaching);
      if_blocks[current_block_type_index].d(detaching);
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
        ctx[6]
      ),
      elem_id: (
        /*elem_id*/
        ctx[4]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[5]
      ),
      padding: (
        /*container*/
        ctx[12]
      ),
      allow_overflow: false,
      scale: (
        /*scale*/
        ctx[13]
      ),
      min_width: (
        /*min_width*/
        ctx[14]
      ),
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
      64)
        block_changes.visible = /*visible*/
        ctx2[6];
      if (dirty[0] & /*elem_id*/
      16)
        block_changes.elem_id = /*elem_id*/
        ctx2[4];
      if (dirty[0] & /*elem_classes*/
      32)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[5];
      if (dirty[0] & /*container*/
      4096)
        block_changes.padding = /*container*/
        ctx2[12];
      if (dirty[0] & /*scale*/
      8192)
        block_changes.scale = /*scale*/
        ctx2[13];
      if (dirty[0] & /*min_width*/
      16384)
        block_changes.min_width = /*min_width*/
        ctx2[14];
      if (dirty[0] & /*choices, max_choices, root, label, info, show_label, allow_custom_value, filterable, container, gradio, interactive, value, value_is_output, multiselect, loading_status*/
      1023887 | dirty[1] & /*$$scope*/
      64) {
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
  let { label = "Dropdown" } = $$props;
  let { info = void 0 } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { multiselect = false } = $$props;
  let { value = multiselect ? [] : void 0 } = $$props;
  let { value_is_output = false } = $$props;
  let { max_choices = null } = $$props;
  let { choices } = $$props;
  let { show_label } = $$props;
  let { filterable } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status } = $$props;
  let { allow_custom_value = false } = $$props;
  let { root } = $$props;
  let { gradio } = $$props;
  let { interactive } = $$props;
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  function multiselect_1_value_binding(value$1) {
    value = value$1;
    $$invalidate(0, value);
  }
  function multiselect_1_value_is_output_binding(value2) {
    value_is_output = value2;
    $$invalidate(1, value_is_output);
  }
  const change_handler = () => gradio.dispatch("change");
  const input_handler = () => gradio.dispatch("input");
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  const blur_handler = () => gradio.dispatch("blur");
  const focus_handler = () => gradio.dispatch("focus");
  const key_up_handler = () => gradio.dispatch("key_up");
  function dropdown_value_binding(value$1) {
    value = value$1;
    $$invalidate(0, value);
  }
  function dropdown_value_is_output_binding(value2) {
    value_is_output = value2;
    $$invalidate(1, value_is_output);
  }
  const change_handler_1 = () => gradio.dispatch("change");
  const input_handler_1 = () => gradio.dispatch("input");
  const select_handler_1 = (e) => gradio.dispatch("select", e.detail);
  const blur_handler_1 = () => gradio.dispatch("blur");
  const focus_handler_1 = () => gradio.dispatch("focus");
  const key_up_handler_1 = (e) => gradio.dispatch("key_up", e.detail);
  $$self.$$set = ($$props2) => {
    if ("label" in $$props2)
      $$invalidate(2, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(3, info = $$props2.info);
    if ("elem_id" in $$props2)
      $$invalidate(4, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(5, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(6, visible = $$props2.visible);
    if ("multiselect" in $$props2)
      $$invalidate(7, multiselect = $$props2.multiselect);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("value_is_output" in $$props2)
      $$invalidate(1, value_is_output = $$props2.value_is_output);
    if ("max_choices" in $$props2)
      $$invalidate(8, max_choices = $$props2.max_choices);
    if ("choices" in $$props2)
      $$invalidate(9, choices = $$props2.choices);
    if ("show_label" in $$props2)
      $$invalidate(10, show_label = $$props2.show_label);
    if ("filterable" in $$props2)
      $$invalidate(11, filterable = $$props2.filterable);
    if ("container" in $$props2)
      $$invalidate(12, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(13, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(14, min_width = $$props2.min_width);
    if ("loading_status" in $$props2)
      $$invalidate(15, loading_status = $$props2.loading_status);
    if ("allow_custom_value" in $$props2)
      $$invalidate(16, allow_custom_value = $$props2.allow_custom_value);
    if ("root" in $$props2)
      $$invalidate(17, root = $$props2.root);
    if ("gradio" in $$props2)
      $$invalidate(18, gradio = $$props2.gradio);
    if ("interactive" in $$props2)
      $$invalidate(19, interactive = $$props2.interactive);
  };
  return [
    value,
    value_is_output,
    label,
    info,
    elem_id,
    elem_classes,
    visible,
    multiselect,
    max_choices,
    choices,
    show_label,
    filterable,
    container,
    scale,
    min_width,
    loading_status,
    allow_custom_value,
    root,
    gradio,
    interactive,
    clear_status_handler,
    multiselect_1_value_binding,
    multiselect_1_value_is_output_binding,
    change_handler,
    input_handler,
    select_handler,
    blur_handler,
    focus_handler,
    key_up_handler,
    dropdown_value_binding,
    dropdown_value_is_output_binding,
    change_handler_1,
    input_handler_1,
    select_handler_1,
    blur_handler_1,
    focus_handler_1,
    key_up_handler_1
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
        label: 2,
        info: 3,
        elem_id: 4,
        elem_classes: 5,
        visible: 6,
        multiselect: 7,
        value: 0,
        value_is_output: 1,
        max_choices: 8,
        choices: 9,
        show_label: 10,
        filterable: 11,
        container: 12,
        scale: 13,
        min_width: 14,
        loading_status: 15,
        allow_custom_value: 16,
        root: 17,
        gradio: 18,
        interactive: 19
      },
      null,
      [-1, -1]
    );
  }
}
export {
  Dropdown as BaseDropdown,
  default2 as BaseExample,
  Multiselect as BaseMultiselect,
  Index as default
};
