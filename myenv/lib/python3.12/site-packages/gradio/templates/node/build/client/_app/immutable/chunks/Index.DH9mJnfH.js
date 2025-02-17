import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, element, space, claim_element, claim_space, toggle_class, set_input_value, listen, group_outros, check_outros, run_all, binding_callbacks, text, claim_text, set_data } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, i as BlockTitle } from "./2.8WKXZUMv.js";
import { default as default2 } from "./Example.CYe6rnxa.js";
function create_fragment$1(ctx) {
  let svg;
  let rect;
  let line0;
  let line1;
  let line2;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      line0 = svg_element("line");
      line1 = svg_element("line");
      line2 = svg_element("line");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        xmlns: true,
        width: true,
        height: true,
        viewBox: true
      });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(rect).forEach(detach);
      line0 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line0).forEach(detach);
      line1 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line1).forEach(detach);
      line2 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line2).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "2");
      attr(rect, "y", "4");
      attr(rect, "width", "20");
      attr(rect, "height", "18");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(rect, "stroke-linecap", "round");
      attr(rect, "stroke-linejoin", "round");
      attr(rect, "fill", "none");
      attr(line0, "x1", "2");
      attr(line0, "y1", "9");
      attr(line0, "x2", "22");
      attr(line0, "y2", "9");
      attr(line0, "stroke", "currentColor");
      attr(line0, "stroke-width", "2");
      attr(line0, "stroke-linecap", "round");
      attr(line0, "stroke-linejoin", "round");
      attr(line0, "fill", "none");
      attr(line1, "x1", "7");
      attr(line1, "y1", "2");
      attr(line1, "x2", "7");
      attr(line1, "y2", "6");
      attr(line1, "stroke", "currentColor");
      attr(line1, "stroke-width", "2");
      attr(line1, "stroke-linecap", "round");
      attr(line1, "stroke-linejoin", "round");
      attr(line1, "fill", "none");
      attr(line2, "x1", "17");
      attr(line2, "y1", "2");
      attr(line2, "x2", "17");
      attr(line2, "y2", "6");
      attr(line2, "stroke", "currentColor");
      attr(line2, "stroke-width", "2");
      attr(line2, "stroke-linecap", "round");
      attr(line2, "stroke-linejoin", "round");
      attr(line2, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "width", "24px");
      attr(svg, "height", "24px");
      attr(svg, "viewBox", "0 0 24 24");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, line0);
      append_hydration(svg, line1);
      append_hydration(svg, line2);
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
class Calendar extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$1, safe_not_equal, {});
  }
}
function create_default_slot_1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[1]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[1]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
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
        detach(t);
      }
    }
  };
}
function create_else_block(ctx) {
  let input;
  let mounted;
  let dispose;
  return {
    c() {
      input = element("input");
      this.h();
    },
    l(nodes) {
      input = claim_element(nodes, "INPUT", { type: true, class: true, step: true });
      this.h();
    },
    h() {
      attr(input, "type", "date");
      attr(input, "class", "datetime svelte-d4qsy2");
      attr(input, "step", "1");
      input.disabled = /*disabled*/
      ctx[16];
    },
    m(target, anchor) {
      insert_hydration(target, input, anchor);
      ctx[26](input);
      set_input_value(
        input,
        /*datevalue*/
        ctx[14]
      );
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler_2*/
            ctx[27]
          ),
          listen(
            input,
            "input",
            /*input_handler_1*/
            ctx[28]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*disabled*/
      65536) {
        input.disabled = /*disabled*/
        ctx2[16];
      }
      if (dirty[0] & /*datevalue*/
      16384) {
        set_input_value(
          input,
          /*datevalue*/
          ctx2[14]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(input);
      }
      ctx[26](null);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_if_block_1(ctx) {
  let input;
  let mounted;
  let dispose;
  return {
    c() {
      input = element("input");
      this.h();
    },
    l(nodes) {
      input = claim_element(nodes, "INPUT", { type: true, class: true, step: true });
      this.h();
    },
    h() {
      attr(input, "type", "datetime-local");
      attr(input, "class", "datetime svelte-d4qsy2");
      attr(input, "step", "1");
      input.disabled = /*disabled*/
      ctx[16];
    },
    m(target, anchor) {
      insert_hydration(target, input, anchor);
      ctx[23](input);
      set_input_value(
        input,
        /*datevalue*/
        ctx[14]
      );
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler_1*/
            ctx[24]
          ),
          listen(
            input,
            "input",
            /*input_handler*/
            ctx[25]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*disabled*/
      65536) {
        input.disabled = /*disabled*/
        ctx2[16];
      }
      if (dirty[0] & /*datevalue*/
      16384) {
        set_input_value(
          input,
          /*datevalue*/
          ctx2[14]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(input);
      }
      ctx[23](null);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_if_block(ctx) {
  let button;
  let calendar;
  let current;
  let mounted;
  let dispose;
  calendar = new Calendar({});
  return {
    c() {
      button = element("button");
      create_component(calendar.$$.fragment);
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      claim_component(calendar.$$.fragment, button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "calendar svelte-d4qsy2");
      button.disabled = /*disabled*/
      ctx[16];
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      mount_component(calendar, button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*click_handler*/
          ctx[29]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (!current || dirty[0] & /*disabled*/
      65536) {
        button.disabled = /*disabled*/
        ctx2[16];
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(calendar.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(calendar.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      destroy_component(calendar);
      mounted = false;
      dispose();
    }
  };
}
function create_default_slot(ctx) {
  let div0;
  let blocktitle;
  let t0;
  let div1;
  let input;
  let t1;
  let t2;
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
        ctx[2]
      ),
      info: (
        /*info*/
        ctx[3]
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  function select_block_type(ctx2, dirty) {
    if (
      /*include_time*/
      ctx2[11]
    )
      return create_if_block_1;
    return create_else_block;
  }
  let current_block_type = select_block_type(ctx);
  let if_block0 = current_block_type(ctx);
  let if_block1 = (
    /*interactive*/
    ctx[4] && create_if_block(ctx)
  );
  return {
    c() {
      div0 = element("div");
      create_component(blocktitle.$$.fragment);
      t0 = space();
      div1 = element("div");
      input = element("input");
      t1 = space();
      if_block0.c();
      t2 = space();
      if (if_block1)
        if_block1.c();
      this.h();
    },
    l(nodes) {
      div0 = claim_element(nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      claim_component(blocktitle.$$.fragment, div0_nodes);
      div0_nodes.forEach(detach);
      t0 = claim_space(nodes);
      div1 = claim_element(nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      input = claim_element(div1_nodes, "INPUT", { class: true });
      t1 = claim_space(div1_nodes);
      if_block0.l(div1_nodes);
      t2 = claim_space(div1_nodes);
      if (if_block1)
        if_block1.l(div1_nodes);
      div1_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div0, "class", "label-content svelte-d4qsy2");
      attr(input, "class", "time svelte-d4qsy2");
      input.disabled = /*disabled*/
      ctx[16];
      toggle_class(input, "invalid", !/*valid*/
      ctx[15]);
      attr(div1, "class", "timebox svelte-d4qsy2");
    },
    m(target, anchor) {
      insert_hydration(target, div0, anchor);
      mount_component(blocktitle, div0, null);
      insert_hydration(target, t0, anchor);
      insert_hydration(target, div1, anchor);
      append_hydration(div1, input);
      set_input_value(
        input,
        /*entered_value*/
        ctx[12]
      );
      append_hydration(div1, t1);
      if_block0.m(div1, null);
      append_hydration(div1, t2);
      if (if_block1)
        if_block1.m(div1, null);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler*/
            ctx[21]
          ),
          listen(
            input,
            "keydown",
            /*keydown_handler*/
            ctx[22]
          ),
          listen(
            input,
            "blur",
            /*submit_values*/
            ctx[18]
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
      4)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[2];
      if (dirty[0] & /*info*/
      8)
        blocktitle_changes.info = /*info*/
        ctx2[3];
      if (dirty[0] & /*label*/
      2 | dirty[1] & /*$$scope*/
      1) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      if (!current || dirty[0] & /*disabled*/
      65536) {
        input.disabled = /*disabled*/
        ctx2[16];
      }
      if (dirty[0] & /*entered_value*/
      4096 && input.value !== /*entered_value*/
      ctx2[12]) {
        set_input_value(
          input,
          /*entered_value*/
          ctx2[12]
        );
      }
      if (!current || dirty[0] & /*valid*/
      32768) {
        toggle_class(input, "invalid", !/*valid*/
        ctx2[15]);
      }
      if (current_block_type === (current_block_type = select_block_type(ctx2)) && if_block0) {
        if_block0.p(ctx2, dirty);
      } else {
        if_block0.d(1);
        if_block0 = current_block_type(ctx2);
        if (if_block0) {
          if_block0.c();
          if_block0.m(div1, t2);
        }
      }
      if (
        /*interactive*/
        ctx2[4]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*interactive*/
          16) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div1, null);
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
      transition_in(blocktitle.$$.fragment, local);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(blocktitle.$$.fragment, local);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div0);
        detach(t0);
        detach(div1);
      }
      destroy_component(blocktitle);
      if_block0.d();
      if (if_block1)
        if_block1.d();
      mounted = false;
      run_all(dispose);
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
        ctx[7]
      ),
      elem_id: (
        /*elem_id*/
        ctx[5]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[6]
      ),
      scale: (
        /*scale*/
        ctx[8]
      ),
      min_width: (
        /*min_width*/
        ctx[9]
      ),
      allow_overflow: false,
      padding: true,
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
      128)
        block_changes.visible = /*visible*/
        ctx2[7];
      if (dirty[0] & /*elem_id*/
      32)
        block_changes.elem_id = /*elem_id*/
        ctx2[5];
      if (dirty[0] & /*elem_classes*/
      64)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[6];
      if (dirty[0] & /*scale*/
      256)
        block_changes.scale = /*scale*/
        ctx2[8];
      if (dirty[0] & /*min_width*/
      512)
        block_changes.min_width = /*min_width*/
        ctx2[9];
      if (dirty[0] & /*disabled, datetime, interactive, datevalue, entered_value, include_time, valid, gradio, root, show_label, info, label*/
      130079 | dirty[1] & /*$$scope*/
      1) {
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
  let disabled;
  let valid;
  let { gradio } = $$props;
  let { label = "Time" } = $$props;
  let { show_label = true } = $$props;
  let { info = void 0 } = $$props;
  let { interactive } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = "" } = $$props;
  let old_value = value;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { root } = $$props;
  let { include_time = true } = $$props;
  const format_date = (date) => {
    if (date.toJSON() === null)
      return "";
    const pad = (num) => num.toString().padStart(2, "0");
    const year = date.getFullYear();
    const month = pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hours = pad(date.getHours());
    const minutes = pad(date.getMinutes());
    const seconds = pad(date.getSeconds());
    const date_str = `${year}-${month}-${day}`;
    const time_str = `${hours}:${minutes}:${seconds}`;
    if (include_time) {
      return `${date_str} ${time_str}`;
    }
    return date_str;
  };
  let entered_value = value;
  let datetime;
  let datevalue = value;
  const date_is_valid_format = (date) => {
    if (date === null || date === "")
      return true;
    const valid_regex = include_time ? /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/ : /^\d{4}-\d{2}-\d{2}$/;
    const is_valid_date = date.match(valid_regex) !== null;
    const is_valid_now = date.match(/^(?:\s*now\s*(?:-\s*\d+\s*[dmhs])?)?\s*$/) !== null;
    return is_valid_date || is_valid_now;
  };
  const submit_values = () => {
    if (entered_value === value)
      return;
    if (!date_is_valid_format(entered_value))
      return;
    $$invalidate(20, old_value = $$invalidate(19, value = entered_value));
    gradio.dispatch("change");
  };
  function input_input_handler() {
    entered_value = this.value;
    $$invalidate(12, entered_value), $$invalidate(19, value), $$invalidate(20, old_value), $$invalidate(0, gradio);
  }
  const keydown_handler = (evt) => {
    if (evt.key === "Enter") {
      submit_values();
      gradio.dispatch("submit");
    }
  };
  function input_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      datetime = $$value;
      $$invalidate(13, datetime);
    });
  }
  function input_input_handler_1() {
    datevalue = this.value;
    $$invalidate(14, datevalue), $$invalidate(19, value), $$invalidate(20, old_value), $$invalidate(0, gradio);
  }
  const input_handler = () => {
    const date = new Date(datevalue);
    $$invalidate(12, entered_value = format_date(date));
    submit_values();
  };
  function input_binding_1($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      datetime = $$value;
      $$invalidate(13, datetime);
    });
  }
  function input_input_handler_2() {
    datevalue = this.value;
    $$invalidate(14, datevalue), $$invalidate(19, value), $$invalidate(20, old_value), $$invalidate(0, gradio);
  }
  const input_handler_1 = () => {
    const date = /* @__PURE__ */ new Date(datevalue + "T00:00:00");
    $$invalidate(12, entered_value = format_date(date));
    submit_values();
  };
  const click_handler = () => {
    datetime.showPicker();
  };
  $$self.$$set = ($$props2) => {
    if ("gradio" in $$props2)
      $$invalidate(0, gradio = $$props2.gradio);
    if ("label" in $$props2)
      $$invalidate(1, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(2, show_label = $$props2.show_label);
    if ("info" in $$props2)
      $$invalidate(3, info = $$props2.info);
    if ("interactive" in $$props2)
      $$invalidate(4, interactive = $$props2.interactive);
    if ("elem_id" in $$props2)
      $$invalidate(5, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(6, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(7, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(19, value = $$props2.value);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(9, min_width = $$props2.min_width);
    if ("root" in $$props2)
      $$invalidate(10, root = $$props2.root);
    if ("include_time" in $$props2)
      $$invalidate(11, include_time = $$props2.include_time);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*interactive*/
    16) {
      $$invalidate(16, disabled = !interactive);
    }
    if ($$self.$$.dirty[0] & /*value, old_value, gradio*/
    1572865) {
      if (value !== old_value) {
        $$invalidate(20, old_value = value);
        $$invalidate(12, entered_value = value);
        $$invalidate(14, datevalue = value);
        gradio.dispatch("change");
      }
    }
    if ($$self.$$.dirty[0] & /*entered_value*/
    4096) {
      $$invalidate(15, valid = date_is_valid_format(entered_value));
    }
  };
  return [
    gradio,
    label,
    show_label,
    info,
    interactive,
    elem_id,
    elem_classes,
    visible,
    scale,
    min_width,
    root,
    include_time,
    entered_value,
    datetime,
    datevalue,
    valid,
    disabled,
    format_date,
    submit_values,
    value,
    old_value,
    input_input_handler,
    keydown_handler,
    input_binding,
    input_input_handler_1,
    input_handler,
    input_binding_1,
    input_input_handler_2,
    input_handler_1,
    click_handler
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
        gradio: 0,
        label: 1,
        show_label: 2,
        info: 3,
        interactive: 4,
        elem_id: 5,
        elem_classes: 6,
        visible: 7,
        value: 19,
        scale: 8,
        min_width: 9,
        root: 10,
        include_time: 11
      },
      null,
      [-1, -1]
    );
  }
}
export {
  default2 as BaseExample,
  Index as default
};
