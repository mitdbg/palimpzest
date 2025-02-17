import { SvelteComponent, init, safe_not_equal, create_component, claim_component, mount_component, listen, transition_in, transition_out, destroy_component, afterUpdate, globals, assign, space, element, text, claim_space, claim_element, children, detach, claim_text, attr, insert_hydration, append_hydration, set_input_value, get_spread_update, get_spread_object, to_number, set_data, run_all, binding_callbacks } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, S as Static, i as BlockTitle } from "./2.8WKXZUMv.js";
const { window: window_1 } = globals;
function create_default_slot_1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[5]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[5]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      32)
        set_data(
          t,
          /*label*/
          ctx2[5]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block(ctx) {
  let button;
  let t;
  let mounted;
  let dispose;
  return {
    c() {
      button = element("button");
      t = text("↺");
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", {
        class: true,
        "aria-label": true,
        "data-testid": true
      });
      var button_nodes = children(button);
      t = claim_text(button_nodes, "↺");
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "reset-button svelte-10lj3xl");
      button.disabled = /*disabled*/
      ctx[19];
      attr(button, "aria-label", "Reset to default value");
      attr(button, "data-testid", "reset-button");
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      append_hydration(button, t);
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*reset_value*/
          ctx[25]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*disabled*/
      524288) {
        button.disabled = /*disabled*/
        ctx2[19];
      }
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
function create_default_slot(ctx) {
  let statustracker;
  let t0;
  let div3;
  let div1;
  let label_1;
  let blocktitle;
  let t1;
  let div0;
  let input0;
  let input0_aria_label_value;
  let t2;
  let t3;
  let div2;
  let span0;
  let t4;
  let t5;
  let input1;
  let input1_aria_label_value;
  let t6;
  let span1;
  let t7;
  let current;
  let mounted;
  let dispose;
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[1].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[1].i18n
    ) },
    /*loading_status*/
    ctx[14]
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
  blocktitle = new BlockTitle({
    props: {
      root: (
        /*root*/
        ctx[15]
      ),
      show_label: (
        /*show_label*/
        ctx[13]
      ),
      info: (
        /*info*/
        ctx[6]
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  let if_block = (
    /*show_reset_button*/
    ctx[16] && create_if_block(ctx)
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t0 = space();
      div3 = element("div");
      div1 = element("div");
      label_1 = element("label");
      create_component(blocktitle.$$.fragment);
      t1 = space();
      div0 = element("div");
      input0 = element("input");
      t2 = space();
      if (if_block)
        if_block.c();
      t3 = space();
      div2 = element("div");
      span0 = element("span");
      t4 = text(
        /*minimum_value*/
        ctx[20]
      );
      t5 = space();
      input1 = element("input");
      t6 = space();
      span1 = element("span");
      t7 = text(
        /*maximum*/
        ctx[11]
      );
      this.h();
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t0 = claim_space(nodes);
      div3 = claim_element(nodes, "DIV", { class: true });
      var div3_nodes = children(div3);
      div1 = claim_element(div3_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      label_1 = claim_element(div1_nodes, "LABEL", { for: true });
      var label_1_nodes = children(label_1);
      claim_component(blocktitle.$$.fragment, label_1_nodes);
      label_1_nodes.forEach(detach);
      t1 = claim_space(div1_nodes);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      input0 = claim_element(div0_nodes, "INPUT", {
        "aria-label": true,
        "data-testid": true,
        type: true,
        min: true,
        max: true,
        step: true,
        class: true
      });
      t2 = claim_space(div0_nodes);
      if (if_block)
        if_block.l(div0_nodes);
      div0_nodes.forEach(detach);
      div1_nodes.forEach(detach);
      t3 = claim_space(div3_nodes);
      div2 = claim_element(div3_nodes, "DIV", { class: true });
      var div2_nodes = children(div2);
      span0 = claim_element(div2_nodes, "SPAN", { class: true });
      var span0_nodes = children(span0);
      t4 = claim_text(
        span0_nodes,
        /*minimum_value*/
        ctx[20]
      );
      span0_nodes.forEach(detach);
      t5 = claim_space(div2_nodes);
      input1 = claim_element(div2_nodes, "INPUT", {
        type: true,
        id: true,
        name: true,
        min: true,
        max: true,
        step: true,
        "aria-label": true,
        class: true
      });
      t6 = claim_space(div2_nodes);
      span1 = claim_element(div2_nodes, "SPAN", { class: true });
      var span1_nodes = children(span1);
      t7 = claim_text(
        span1_nodes,
        /*maximum*/
        ctx[11]
      );
      span1_nodes.forEach(detach);
      div2_nodes.forEach(detach);
      div3_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(
        label_1,
        "for",
        /*id*/
        ctx[21]
      );
      attr(input0, "aria-label", input0_aria_label_value = `number input for ${/*label*/
      ctx[5]}`);
      attr(input0, "data-testid", "number-input");
      attr(input0, "type", "number");
      attr(
        input0,
        "min",
        /*minimum*/
        ctx[10]
      );
      attr(
        input0,
        "max",
        /*maximum*/
        ctx[11]
      );
      attr(
        input0,
        "step",
        /*step*/
        ctx[12]
      );
      input0.disabled = /*disabled*/
      ctx[19];
      attr(input0, "class", "svelte-10lj3xl");
      attr(div0, "class", "tab-like-container svelte-10lj3xl");
      attr(div1, "class", "head svelte-10lj3xl");
      attr(span0, "class", "min_value svelte-10lj3xl");
      attr(input1, "type", "range");
      attr(
        input1,
        "id",
        /*id*/
        ctx[21]
      );
      attr(input1, "name", "cowbell");
      attr(
        input1,
        "min",
        /*minimum*/
        ctx[10]
      );
      attr(
        input1,
        "max",
        /*maximum*/
        ctx[11]
      );
      attr(
        input1,
        "step",
        /*step*/
        ctx[12]
      );
      input1.disabled = /*disabled*/
      ctx[19];
      attr(input1, "aria-label", input1_aria_label_value = `range slider for ${/*label*/
      ctx[5]}`);
      attr(input1, "class", "svelte-10lj3xl");
      attr(span1, "class", "max_value svelte-10lj3xl");
      attr(div2, "class", "slider_input_container svelte-10lj3xl");
      attr(div3, "class", "wrap svelte-10lj3xl");
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t0, anchor);
      insert_hydration(target, div3, anchor);
      append_hydration(div3, div1);
      append_hydration(div1, label_1);
      mount_component(blocktitle, label_1, null);
      append_hydration(div1, t1);
      append_hydration(div1, div0);
      append_hydration(div0, input0);
      set_input_value(
        input0,
        /*value*/
        ctx[0]
      );
      ctx[30](input0);
      append_hydration(div0, t2);
      if (if_block)
        if_block.m(div0, null);
      append_hydration(div3, t3);
      append_hydration(div3, div2);
      append_hydration(div2, span0);
      append_hydration(span0, t4);
      append_hydration(div2, t5);
      append_hydration(div2, input1);
      set_input_value(
        input1,
        /*value*/
        ctx[0]
      );
      ctx[32](input1);
      append_hydration(div2, t6);
      append_hydration(div2, span1);
      append_hydration(span1, t7);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            input0,
            "input",
            /*input0_input_handler*/
            ctx[29]
          ),
          listen(
            input0,
            "blur",
            /*clamp*/
            ctx[23]
          ),
          listen(
            input0,
            "pointerup",
            /*handle_release*/
            ctx[22]
          ),
          listen(
            input1,
            "change",
            /*input1_change_input_handler*/
            ctx[31]
          ),
          listen(
            input1,
            "input",
            /*input1_change_input_handler*/
            ctx[31]
          ),
          listen(
            input1,
            "pointerup",
            /*handle_release*/
            ctx[22]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      16386 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        2 && { autoscroll: (
          /*gradio*/
          ctx2[1].autoscroll
        ) },
        dirty[0] & /*gradio*/
        2 && { i18n: (
          /*gradio*/
          ctx2[1].i18n
        ) },
        dirty[0] & /*loading_status*/
        16384 && get_spread_object(
          /*loading_status*/
          ctx2[14]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const blocktitle_changes = {};
      if (dirty[0] & /*root*/
      32768)
        blocktitle_changes.root = /*root*/
        ctx2[15];
      if (dirty[0] & /*show_label*/
      8192)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[13];
      if (dirty[0] & /*info*/
      64)
        blocktitle_changes.info = /*info*/
        ctx2[6];
      if (dirty[0] & /*label*/
      32 | dirty[1] & /*$$scope*/
      128) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      if (!current || dirty[0] & /*label*/
      32 && input0_aria_label_value !== (input0_aria_label_value = `number input for ${/*label*/
      ctx2[5]}`)) {
        attr(input0, "aria-label", input0_aria_label_value);
      }
      if (!current || dirty[0] & /*minimum*/
      1024) {
        attr(
          input0,
          "min",
          /*minimum*/
          ctx2[10]
        );
      }
      if (!current || dirty[0] & /*maximum*/
      2048) {
        attr(
          input0,
          "max",
          /*maximum*/
          ctx2[11]
        );
      }
      if (!current || dirty[0] & /*step*/
      4096) {
        attr(
          input0,
          "step",
          /*step*/
          ctx2[12]
        );
      }
      if (!current || dirty[0] & /*disabled*/
      524288) {
        input0.disabled = /*disabled*/
        ctx2[19];
      }
      if (dirty[0] & /*value*/
      1 && to_number(input0.value) !== /*value*/
      ctx2[0]) {
        set_input_value(
          input0,
          /*value*/
          ctx2[0]
        );
      }
      if (
        /*show_reset_button*/
        ctx2[16]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          if_block.m(div0, null);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (!current || dirty[0] & /*minimum_value*/
      1048576)
        set_data(
          t4,
          /*minimum_value*/
          ctx2[20]
        );
      if (!current || dirty[0] & /*minimum*/
      1024) {
        attr(
          input1,
          "min",
          /*minimum*/
          ctx2[10]
        );
      }
      if (!current || dirty[0] & /*maximum*/
      2048) {
        attr(
          input1,
          "max",
          /*maximum*/
          ctx2[11]
        );
      }
      if (!current || dirty[0] & /*step*/
      4096) {
        attr(
          input1,
          "step",
          /*step*/
          ctx2[12]
        );
      }
      if (!current || dirty[0] & /*disabled*/
      524288) {
        input1.disabled = /*disabled*/
        ctx2[19];
      }
      if (!current || dirty[0] & /*label*/
      32 && input1_aria_label_value !== (input1_aria_label_value = `range slider for ${/*label*/
      ctx2[5]}`)) {
        attr(input1, "aria-label", input1_aria_label_value);
      }
      if (dirty[0] & /*value*/
      1) {
        set_input_value(
          input1,
          /*value*/
          ctx2[0]
        );
      }
      if (!current || dirty[0] & /*maximum*/
      2048)
        set_data(
          t7,
          /*maximum*/
          ctx2[11]
        );
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(blocktitle.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(blocktitle.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(div3);
      }
      destroy_component(statustracker, detaching);
      destroy_component(blocktitle);
      ctx[30](null);
      if (if_block)
        if_block.d();
      ctx[32](null);
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let current;
  let mounted;
  let dispose;
  block = new Block({
    props: {
      visible: (
        /*visible*/
        ctx[4]
      ),
      elem_id: (
        /*elem_id*/
        ctx[2]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[3]
      ),
      container: (
        /*container*/
        ctx[7]
      ),
      scale: (
        /*scale*/
        ctx[8]
      ),
      min_width: (
        /*min_width*/
        ctx[9]
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
      if (!mounted) {
        dispose = listen(
          window_1,
          "resize",
          /*handle_resize*/
          ctx[24]
        );
        mounted = true;
      }
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
      if (dirty[0] & /*container*/
      128)
        block_changes.container = /*container*/
        ctx2[7];
      if (dirty[0] & /*scale*/
      256)
        block_changes.scale = /*scale*/
        ctx2[8];
      if (dirty[0] & /*min_width*/
      512)
        block_changes.min_width = /*min_width*/
        ctx2[9];
      if (dirty[0] & /*maximum, minimum, step, disabled, label, value, range_input, minimum_value, show_reset_button, number_input, root, show_label, info, gradio, loading_status*/
      2096227 | dirty[1] & /*$$scope*/
      128) {
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
      mounted = false;
      dispose();
    }
  };
}
let _id = 0;
function instance($$self, $$props, $$invalidate) {
  let minimum_value;
  let disabled;
  let { gradio } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = 0 } = $$props;
  let initial_value = value;
  let { label = gradio.i18n("slider.slider") } = $$props;
  let { info = void 0 } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { minimum } = $$props;
  let { maximum = 100 } = $$props;
  let { step } = $$props;
  let { show_label } = $$props;
  let { interactive } = $$props;
  let { loading_status } = $$props;
  let { value_is_output = false } = $$props;
  let { root } = $$props;
  let { show_reset_button } = $$props;
  let range_input;
  let number_input;
  const id = `range_id_${_id++}`;
  function handle_change() {
    gradio.dispatch("change");
    if (!value_is_output) {
      gradio.dispatch("input");
    }
  }
  afterUpdate(() => {
    $$invalidate(26, value_is_output = false);
    set_slider();
  });
  function handle_release(e) {
    gradio.dispatch("release", value);
  }
  function clamp() {
    gradio.dispatch("release", value);
    $$invalidate(0, value = Math.min(Math.max(value, minimum), maximum));
  }
  function set_slider() {
    set_slider_range();
    range_input.addEventListener("input", set_slider_range);
    number_input.addEventListener("input", set_slider_range);
  }
  function set_slider_range() {
    const range = range_input;
    const min = Number(range.min);
    const max = Number(range.max);
    const val = Number(range.value);
    const percentage = (val - min) / (max - min) * 100;
    range.style.setProperty("--range_progress", `${percentage}%`);
  }
  function handle_resize() {
  }
  function reset_value() {
    $$invalidate(0, value = initial_value);
    set_slider_range();
    gradio.dispatch("change");
    gradio.dispatch("release", value);
  }
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  function input0_input_handler() {
    value = to_number(this.value);
    $$invalidate(0, value);
  }
  function input0_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      number_input = $$value;
      $$invalidate(18, number_input);
    });
  }
  function input1_change_input_handler() {
    value = to_number(this.value);
    $$invalidate(0, value);
  }
  function input1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      range_input = $$value;
      $$invalidate(17, range_input);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("gradio" in $$props2)
      $$invalidate(1, gradio = $$props2.gradio);
    if ("elem_id" in $$props2)
      $$invalidate(2, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(3, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(4, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("label" in $$props2)
      $$invalidate(5, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(6, info = $$props2.info);
    if ("container" in $$props2)
      $$invalidate(7, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(9, min_width = $$props2.min_width);
    if ("minimum" in $$props2)
      $$invalidate(10, minimum = $$props2.minimum);
    if ("maximum" in $$props2)
      $$invalidate(11, maximum = $$props2.maximum);
    if ("step" in $$props2)
      $$invalidate(12, step = $$props2.step);
    if ("show_label" in $$props2)
      $$invalidate(13, show_label = $$props2.show_label);
    if ("interactive" in $$props2)
      $$invalidate(27, interactive = $$props2.interactive);
    if ("loading_status" in $$props2)
      $$invalidate(14, loading_status = $$props2.loading_status);
    if ("value_is_output" in $$props2)
      $$invalidate(26, value_is_output = $$props2.value_is_output);
    if ("root" in $$props2)
      $$invalidate(15, root = $$props2.root);
    if ("show_reset_button" in $$props2)
      $$invalidate(16, show_reset_button = $$props2.show_reset_button);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*minimum*/
    1024) {
      $$invalidate(20, minimum_value = minimum ?? 0);
    }
    if ($$self.$$.dirty[0] & /*interactive*/
    134217728) {
      $$invalidate(19, disabled = !interactive);
    }
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      handle_change();
    }
  };
  return [
    value,
    gradio,
    elem_id,
    elem_classes,
    visible,
    label,
    info,
    container,
    scale,
    min_width,
    minimum,
    maximum,
    step,
    show_label,
    loading_status,
    root,
    show_reset_button,
    range_input,
    number_input,
    disabled,
    minimum_value,
    id,
    handle_release,
    clamp,
    handle_resize,
    reset_value,
    value_is_output,
    interactive,
    clear_status_handler,
    input0_input_handler,
    input0_binding,
    input1_change_input_handler,
    input1_binding
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
        gradio: 1,
        elem_id: 2,
        elem_classes: 3,
        visible: 4,
        value: 0,
        label: 5,
        info: 6,
        container: 7,
        scale: 8,
        min_width: 9,
        minimum: 10,
        maximum: 11,
        step: 12,
        show_label: 13,
        interactive: 27,
        loading_status: 14,
        value_is_output: 26,
        root: 15,
        show_reset_button: 16
      },
      null,
      [-1, -1]
    );
  }
}
export {
  Index as default
};
