import { SvelteComponent, init, safe_not_equal, element, space, claim_element, children, claim_space, detach, attr, insert_hydration, append_hydration, noop, createEventDispatcher, text, claim_text, toggle_class, set_style, set_data, ensure_array_like, empty, destroy_each, listen, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, assign, get_spread_update, get_spread_object, group_outros, check_outros } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, S as Static } from "./2.8WKXZUMv.js";
import { L as LineChart } from "./LineChart.D-OPS8mj.js";
import { B as BlockLabel } from "./BlockLabel.BcwqJKEo.js";
import { E as Empty } from "./Empty.DwcIP_BA.js";
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[6] = list[i];
  child_ctx[8] = i;
  return child_ctx;
}
function create_if_block_1$1(ctx) {
  let h2;
  let t_value = (
    /*value*/
    ctx[0].label + ""
  );
  let t;
  return {
    c() {
      h2 = element("h2");
      t = text(t_value);
      this.h();
    },
    l(nodes) {
      h2 = claim_element(nodes, "H2", { class: true, "data-testid": true });
      var h2_nodes = children(h2);
      t = claim_text(h2_nodes, t_value);
      h2_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(h2, "class", "output-class svelte-1mutzus");
      attr(h2, "data-testid", "label-output-value");
      toggle_class(h2, "no-confidence", !("confidences" in /*value*/
      ctx[0]));
      set_style(
        h2,
        "background-color",
        /*color*/
        ctx[1] || "transparent"
      );
    },
    m(target, anchor) {
      insert_hydration(target, h2, anchor);
      append_hydration(h2, t);
    },
    p(ctx2, dirty) {
      if (dirty & /*value*/
      1 && t_value !== (t_value = /*value*/
      ctx2[0].label + ""))
        set_data(t, t_value);
      if (dirty & /*value*/
      1) {
        toggle_class(h2, "no-confidence", !("confidences" in /*value*/
        ctx2[0]));
      }
      if (dirty & /*color*/
      2) {
        set_style(
          h2,
          "background-color",
          /*color*/
          ctx2[1] || "transparent"
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(h2);
      }
    }
  };
}
function create_if_block$1(ctx) {
  let each_1_anchor;
  let each_value = ensure_array_like(
    /*value*/
    ctx[0].confidences
  );
  let each_blocks = [];
  for (let i = 0; i < each_value.length; i += 1) {
    each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
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
    },
    p(ctx2, dirty) {
      if (dirty & /*value, selectable, dispatch, Math, get_aria_referenceable_id*/
      21) {
        each_value = ensure_array_like(
          /*value*/
          ctx2[0].confidences
        );
        let i;
        for (i = 0; i < each_value.length; i += 1) {
          const child_ctx = get_each_context(ctx2, each_value, i);
          if (each_blocks[i]) {
            each_blocks[i].p(child_ctx, dirty);
          } else {
            each_blocks[i] = create_each_block(child_ctx);
            each_blocks[i].c();
            each_blocks[i].m(each_1_anchor.parentNode, each_1_anchor);
          }
        }
        for (; i < each_blocks.length; i += 1) {
          each_blocks[i].d(1);
        }
        each_blocks.length = each_value.length;
      }
    },
    d(detaching) {
      if (detaching) {
        detach(each_1_anchor);
      }
      destroy_each(each_blocks, detaching);
    }
  };
}
function create_each_block(ctx) {
  let button;
  let div1;
  let meter;
  let meter_aria_labelledby_value;
  let meter_aria_label_value;
  let meter_aria_valuenow_value;
  let meter_value_value;
  let t0;
  let dl;
  let dt;
  let t1_value = (
    /*confidence_set*/
    ctx[6].label + ""
  );
  let t1;
  let t2;
  let dt_id_value;
  let div0;
  let dd;
  let t3_value = Math.round(
    /*confidence_set*/
    ctx[6].confidence * 100
  ) + "";
  let t3;
  let t4;
  let t5;
  let button_data_testid_value;
  let mounted;
  let dispose;
  function click_handler() {
    return (
      /*click_handler*/
      ctx[5](
        /*i*/
        ctx[8],
        /*confidence_set*/
        ctx[6]
      )
    );
  }
  return {
    c() {
      button = element("button");
      div1 = element("div");
      meter = element("meter");
      t0 = space();
      dl = element("dl");
      dt = element("dt");
      t1 = text(t1_value);
      t2 = space();
      div0 = element("div");
      dd = element("dd");
      t3 = text(t3_value);
      t4 = text("%");
      t5 = space();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true, "data-testid": true });
      var button_nodes = children(button);
      div1 = claim_element(button_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      meter = claim_element(div1_nodes, "METER", {
        "aria-labelledby": true,
        "aria-label": true,
        "aria-valuenow": true,
        "aria-valuemin": true,
        "aria-valuemax": true,
        class: true,
        min: true,
        max: true,
        style: true
      });
      children(meter).forEach(detach);
      t0 = claim_space(div1_nodes);
      dl = claim_element(div1_nodes, "DL", { class: true });
      var dl_nodes = children(dl);
      dt = claim_element(dl_nodes, "DT", { id: true, class: true });
      var dt_nodes = children(dt);
      t1 = claim_text(dt_nodes, t1_value);
      t2 = claim_space(dt_nodes);
      dt_nodes.forEach(detach);
      div0 = claim_element(dl_nodes, "DIV", { class: true });
      children(div0).forEach(detach);
      dd = claim_element(dl_nodes, "DD", { class: true });
      var dd_nodes = children(dd);
      t3 = claim_text(dd_nodes, t3_value);
      t4 = claim_text(dd_nodes, "%");
      dd_nodes.forEach(detach);
      dl_nodes.forEach(detach);
      div1_nodes.forEach(detach);
      t5 = claim_space(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(meter, "aria-labelledby", meter_aria_labelledby_value = get_aria_referenceable_id(`meter-text-${/*confidence_set*/
      ctx[6].label}`));
      attr(meter, "aria-label", meter_aria_label_value = /*confidence_set*/
      ctx[6].label);
      attr(meter, "aria-valuenow", meter_aria_valuenow_value = Math.round(
        /*confidence_set*/
        ctx[6].confidence * 100
      ));
      attr(meter, "aria-valuemin", "0");
      attr(meter, "aria-valuemax", "100");
      attr(meter, "class", "bar svelte-1mutzus");
      attr(meter, "min", "0");
      attr(meter, "max", "1");
      meter.value = meter_value_value = /*confidence_set*/
      ctx[6].confidence;
      set_style(
        meter,
        "width",
        /*confidence_set*/
        ctx[6].confidence * 100 + "%"
      );
      set_style(meter, "background", "var(--stat-background-fill)");
      attr(dt, "id", dt_id_value = get_aria_referenceable_id(`meter-text-${/*confidence_set*/
      ctx[6].label}`));
      attr(dt, "class", "text svelte-1mutzus");
      attr(div0, "class", "line svelte-1mutzus");
      attr(dd, "class", "confidence svelte-1mutzus");
      attr(dl, "class", "label svelte-1mutzus");
      attr(div1, "class", "inner-wrap svelte-1mutzus");
      attr(button, "class", "confidence-set group svelte-1mutzus");
      attr(button, "data-testid", button_data_testid_value = `${/*confidence_set*/
      ctx[6].label}-confidence-set`);
      toggle_class(
        button,
        "selectable",
        /*selectable*/
        ctx[2]
      );
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      append_hydration(button, div1);
      append_hydration(div1, meter);
      append_hydration(div1, t0);
      append_hydration(div1, dl);
      append_hydration(dl, dt);
      append_hydration(dt, t1);
      append_hydration(dt, t2);
      append_hydration(dl, div0);
      append_hydration(dl, dd);
      append_hydration(dd, t3);
      append_hydration(dd, t4);
      append_hydration(button, t5);
      if (!mounted) {
        dispose = listen(button, "click", click_handler);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (dirty & /*value*/
      1 && meter_aria_labelledby_value !== (meter_aria_labelledby_value = get_aria_referenceable_id(`meter-text-${/*confidence_set*/
      ctx[6].label}`))) {
        attr(meter, "aria-labelledby", meter_aria_labelledby_value);
      }
      if (dirty & /*value*/
      1 && meter_aria_label_value !== (meter_aria_label_value = /*confidence_set*/
      ctx[6].label)) {
        attr(meter, "aria-label", meter_aria_label_value);
      }
      if (dirty & /*value*/
      1 && meter_aria_valuenow_value !== (meter_aria_valuenow_value = Math.round(
        /*confidence_set*/
        ctx[6].confidence * 100
      ))) {
        attr(meter, "aria-valuenow", meter_aria_valuenow_value);
      }
      if (dirty & /*value*/
      1 && meter_value_value !== (meter_value_value = /*confidence_set*/
      ctx[6].confidence)) {
        meter.value = meter_value_value;
      }
      if (dirty & /*value*/
      1) {
        set_style(
          meter,
          "width",
          /*confidence_set*/
          ctx[6].confidence * 100 + "%"
        );
      }
      if (dirty & /*value*/
      1 && t1_value !== (t1_value = /*confidence_set*/
      ctx[6].label + ""))
        set_data(t1, t1_value);
      if (dirty & /*value*/
      1 && dt_id_value !== (dt_id_value = get_aria_referenceable_id(`meter-text-${/*confidence_set*/
      ctx[6].label}`))) {
        attr(dt, "id", dt_id_value);
      }
      if (dirty & /*value*/
      1 && t3_value !== (t3_value = Math.round(
        /*confidence_set*/
        ctx[6].confidence * 100
      ) + ""))
        set_data(t3, t3_value);
      if (dirty & /*value*/
      1 && button_data_testid_value !== (button_data_testid_value = `${/*confidence_set*/
      ctx[6].label}-confidence-set`)) {
        attr(button, "data-testid", button_data_testid_value);
      }
      if (dirty & /*selectable*/
      4) {
        toggle_class(
          button,
          "selectable",
          /*selectable*/
          ctx[2]
        );
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
function create_fragment$1(ctx) {
  let div;
  let t;
  let if_block0 = (
    /*show_heading*/
    (ctx[3] || !/*value*/
    ctx[0].confidences) && create_if_block_1$1(ctx)
  );
  let if_block1 = typeof /*value*/
  ctx[0] === "object" && /*value*/
  ctx[0].confidences && create_if_block$1(ctx);
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
      attr(div, "class", "container svelte-1mutzus");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      if (if_block0)
        if_block0.m(div, null);
      append_hydration(div, t);
      if (if_block1)
        if_block1.m(div, null);
    },
    p(ctx2, [dirty]) {
      if (
        /*show_heading*/
        ctx2[3] || !/*value*/
        ctx2[0].confidences
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
        } else {
          if_block0 = create_if_block_1$1(ctx2);
          if_block0.c();
          if_block0.m(div, t);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      if (typeof /*value*/
      ctx2[0] === "object" && /*value*/
      ctx2[0].confidences) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
        } else {
          if_block1 = create_if_block$1(ctx2);
          if_block1.c();
          if_block1.m(div, null);
        }
      } else if (if_block1) {
        if_block1.d(1);
        if_block1 = null;
      }
    },
    i: noop,
    o: noop,
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
function get_aria_referenceable_id(elem_id) {
  return elem_id.replace(/\s/g, "-");
}
function instance$1($$self, $$props, $$invalidate) {
  let { value } = $$props;
  const dispatch = createEventDispatcher();
  let { color = void 0 } = $$props;
  let { selectable = false } = $$props;
  let { show_heading = true } = $$props;
  const click_handler = (i, confidence_set) => {
    dispatch("select", { index: i, value: confidence_set.label });
  };
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("color" in $$props2)
      $$invalidate(1, color = $$props2.color);
    if ("selectable" in $$props2)
      $$invalidate(2, selectable = $$props2.selectable);
    if ("show_heading" in $$props2)
      $$invalidate(3, show_heading = $$props2.show_heading);
  };
  return [value, color, selectable, show_heading, dispatch, click_handler];
}
class Label extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$1, create_fragment$1, safe_not_equal, {
      value: 0,
      color: 1,
      selectable: 2,
      show_heading: 3
    });
  }
}
const Label$1 = Label;
function create_if_block_1(ctx) {
  let blocklabel;
  let current;
  blocklabel = new BlockLabel({
    props: {
      Icon: LineChart,
      label: (
        /*label*/
        ctx[6]
      ),
      disable: (
        /*container*/
        ctx[7] === false
      ),
      float: (
        /*show_heading*/
        ctx[13] === true
      )
    }
  });
  return {
    c() {
      create_component(blocklabel.$$.fragment);
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const blocklabel_changes = {};
      if (dirty & /*label*/
      64)
        blocklabel_changes.label = /*label*/
        ctx2[6];
      if (dirty & /*container*/
      128)
        blocklabel_changes.disable = /*container*/
        ctx2[7] === false;
      if (dirty & /*show_heading*/
      8192)
        blocklabel_changes.float = /*show_heading*/
        ctx2[13] === true;
      blocklabel.$set(blocklabel_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(blocklabel.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(blocklabel, detaching);
    }
  };
}
function create_else_block(ctx) {
  let empty_1;
  let current;
  empty_1 = new Empty({
    props: {
      unpadded_box: true,
      $$slots: { default: [create_default_slot_1] },
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
      262144) {
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
function create_if_block(ctx) {
  let label_1;
  let current;
  label_1 = new Label$1({
    props: {
      selectable: (
        /*_selectable*/
        ctx[12]
      ),
      value: (
        /*value*/
        ctx[5]
      ),
      color: (
        /*color*/
        ctx[4]
      ),
      show_heading: (
        /*show_heading*/
        ctx[13]
      )
    }
  });
  label_1.$on(
    "select",
    /*select_handler*/
    ctx[17]
  );
  return {
    c() {
      create_component(label_1.$$.fragment);
    },
    l(nodes) {
      claim_component(label_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(label_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const label_1_changes = {};
      if (dirty & /*_selectable*/
      4096)
        label_1_changes.selectable = /*_selectable*/
        ctx2[12];
      if (dirty & /*value*/
      32)
        label_1_changes.value = /*value*/
        ctx2[5];
      if (dirty & /*color*/
      16)
        label_1_changes.color = /*color*/
        ctx2[4];
      if (dirty & /*show_heading*/
      8192)
        label_1_changes.show_heading = /*show_heading*/
        ctx2[13];
      label_1.$set(label_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(label_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(label_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(label_1, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let labelicon;
  let current;
  labelicon = new LineChart({});
  return {
    c() {
      create_component(labelicon.$$.fragment);
    },
    l(nodes) {
      claim_component(labelicon.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(labelicon, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(labelicon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(labelicon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(labelicon, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let statustracker;
  let t0;
  let t1;
  let current_block_type_index;
  let if_block1;
  let if_block1_anchor;
  let current;
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[0].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[0].i18n
    ) },
    /*loading_status*/
    ctx[10]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[16]
  );
  let if_block0 = (
    /*show_label*/
    ctx[11] && create_if_block_1(ctx)
  );
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*_label*/
      ctx2[14] !== void 0 && /*_label*/
      ctx2[14] !== null
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block1 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t0 = space();
      if (if_block0)
        if_block0.c();
      t1 = space();
      if_block1.c();
      if_block1_anchor = empty();
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t0 = claim_space(nodes);
      if (if_block0)
        if_block0.l(nodes);
      t1 = claim_space(nodes);
      if_block1.l(nodes);
      if_block1_anchor = empty();
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t0, anchor);
      if (if_block0)
        if_block0.m(target, anchor);
      insert_hydration(target, t1, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block1_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty & /*gradio, loading_status*/
      1025 ? get_spread_update(statustracker_spread_levels, [
        dirty & /*gradio*/
        1 && { autoscroll: (
          /*gradio*/
          ctx2[0].autoscroll
        ) },
        dirty & /*gradio*/
        1 && { i18n: (
          /*gradio*/
          ctx2[0].i18n
        ) },
        dirty & /*loading_status*/
        1024 && get_spread_object(
          /*loading_status*/
          ctx2[10]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      if (
        /*show_label*/
        ctx2[11]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty & /*show_label*/
          2048) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_1(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(t1.parentNode, t1);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
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
        if_block1.m(if_block1_anchor.parentNode, if_block1_anchor);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(t1);
        detach(if_block1_anchor);
      }
      destroy_component(statustracker, detaching);
      if (if_block0)
        if_block0.d(detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      test_id: "label",
      visible: (
        /*visible*/
        ctx[3]
      ),
      elem_id: (
        /*elem_id*/
        ctx[1]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[2]
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
      padding: false,
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
    p(ctx2, [dirty]) {
      const block_changes = {};
      if (dirty & /*visible*/
      8)
        block_changes.visible = /*visible*/
        ctx2[3];
      if (dirty & /*elem_id*/
      2)
        block_changes.elem_id = /*elem_id*/
        ctx2[1];
      if (dirty & /*elem_classes*/
      4)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[2];
      if (dirty & /*container*/
      128)
        block_changes.container = /*container*/
        ctx2[7];
      if (dirty & /*scale*/
      256)
        block_changes.scale = /*scale*/
        ctx2[8];
      if (dirty & /*min_width*/
      512)
        block_changes.min_width = /*min_width*/
        ctx2[9];
      if (dirty & /*$$scope, _selectable, value, color, show_heading, gradio, _label, label, container, show_label, loading_status*/
      294129) {
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
  let _label;
  let { gradio } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { color = void 0 } = $$props;
  let { value = {} } = $$props;
  let old_value = null;
  let { label = gradio.i18n("label.label") } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status } = $$props;
  let { show_label = true } = $$props;
  let { _selectable = false } = $$props;
  let { show_heading = true } = $$props;
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const select_handler = ({ detail }) => gradio.dispatch("select", detail);
  $$self.$$set = ($$props2) => {
    if ("gradio" in $$props2)
      $$invalidate(0, gradio = $$props2.gradio);
    if ("elem_id" in $$props2)
      $$invalidate(1, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(2, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(3, visible = $$props2.visible);
    if ("color" in $$props2)
      $$invalidate(4, color = $$props2.color);
    if ("value" in $$props2)
      $$invalidate(5, value = $$props2.value);
    if ("label" in $$props2)
      $$invalidate(6, label = $$props2.label);
    if ("container" in $$props2)
      $$invalidate(7, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(9, min_width = $$props2.min_width);
    if ("loading_status" in $$props2)
      $$invalidate(10, loading_status = $$props2.loading_status);
    if ("show_label" in $$props2)
      $$invalidate(11, show_label = $$props2.show_label);
    if ("_selectable" in $$props2)
      $$invalidate(12, _selectable = $$props2._selectable);
    if ("show_heading" in $$props2)
      $$invalidate(13, show_heading = $$props2.show_heading);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value, old_value, gradio*/
    32801) {
      {
        if (JSON.stringify(value) !== JSON.stringify(old_value)) {
          $$invalidate(15, old_value = value);
          gradio.dispatch("change");
        }
      }
    }
    if ($$self.$$.dirty & /*value*/
    32) {
      $$invalidate(14, _label = value.label);
    }
  };
  return [
    gradio,
    elem_id,
    elem_classes,
    visible,
    color,
    value,
    label,
    container,
    scale,
    min_width,
    loading_status,
    show_label,
    _selectable,
    show_heading,
    _label,
    old_value,
    clear_status_handler,
    select_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      gradio: 0,
      elem_id: 1,
      elem_classes: 2,
      visible: 3,
      color: 4,
      value: 5,
      label: 6,
      container: 7,
      scale: 8,
      min_width: 9,
      loading_status: 10,
      show_label: 11,
      _selectable: 12,
      show_heading: 13
    });
  }
}
export {
  Label$1 as BaseLabel,
  Index as default
};
