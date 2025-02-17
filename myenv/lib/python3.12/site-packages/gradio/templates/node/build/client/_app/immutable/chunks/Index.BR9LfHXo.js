import { SvelteComponent, init, safe_not_equal, element, HtmlTagHydration, claim_element, children, claim_html_tag, detach, attr, toggle_class, insert_hydration, listen, noop, createEventDispatcher, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, assign, space, claim_space, set_style, group_outros, check_outros, get_spread_update, get_spread_object } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, S as Static, z as css_units } from "./2.8WKXZUMv.js";
import { C as Code } from "./Code.VKlWzMBL.js";
import { B as BlockLabel } from "./BlockLabel.BcwqJKEo.js";
function create_fragment$1(ctx) {
  let div;
  let html_tag;
  let div_class_value;
  let mounted;
  let dispose;
  return {
    c() {
      div = element("div");
      html_tag = new HtmlTagHydration(false);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      html_tag = claim_html_tag(div_nodes, false);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      html_tag.a = null;
      attr(div, "class", div_class_value = "prose " + /*elem_classes*/
      ctx[0].join(" ") + " svelte-ydeks8");
      toggle_class(div, "hide", !/*visible*/
      ctx[2]);
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      html_tag.m(
        /*value*/
        ctx[1],
        div
      );
      if (!mounted) {
        dispose = listen(
          div,
          "click",
          /*click_handler*/
          ctx[4]
        );
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (dirty & /*value*/
      2)
        html_tag.p(
          /*value*/
          ctx2[1]
        );
      if (dirty & /*elem_classes*/
      1 && div_class_value !== (div_class_value = "prose " + /*elem_classes*/
      ctx2[0].join(" ") + " svelte-ydeks8")) {
        attr(div, "class", div_class_value);
      }
      if (dirty & /*elem_classes, visible*/
      5) {
        toggle_class(div, "hide", !/*visible*/
        ctx2[2]);
      }
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      mounted = false;
      dispose();
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let { elem_classes = [] } = $$props;
  let { value } = $$props;
  let { visible = true } = $$props;
  const dispatch = createEventDispatcher();
  const click_handler = () => dispatch("click");
  $$self.$$set = ($$props2) => {
    if ("elem_classes" in $$props2)
      $$invalidate(0, elem_classes = $$props2.elem_classes);
    if ("value" in $$props2)
      $$invalidate(1, value = $$props2.value);
    if ("visible" in $$props2)
      $$invalidate(2, visible = $$props2.visible);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value*/
    2) {
      dispatch("change");
    }
  };
  return [elem_classes, value, visible, dispatch, click_handler];
}
class HTML extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$1, create_fragment$1, safe_not_equal, { elem_classes: 0, value: 1, visible: 2 });
  }
}
function create_if_block(ctx) {
  let blocklabel;
  let current;
  blocklabel = new BlockLabel({
    props: {
      Icon: Code,
      show_label: (
        /*show_label*/
        ctx[7]
      ),
      label: (
        /*label*/
        ctx[0]
      ),
      float: false
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
      if (dirty & /*show_label*/
      128)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[7];
      if (dirty & /*label*/
      1)
        blocklabel_changes.label = /*label*/
        ctx2[0];
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
function create_default_slot(ctx) {
  let t0;
  let statustracker;
  let t1;
  let div;
  let html;
  let current;
  let if_block = (
    /*show_label*/
    ctx[7] && create_if_block(ctx)
  );
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[6].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[6].i18n
    ) },
    /*loading_status*/
    ctx[5],
    { variant: "center" }
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[12]
  );
  html = new HTML({
    props: {
      value: (
        /*value*/
        ctx[4]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[2]
      ),
      visible: (
        /*visible*/
        ctx[3]
      )
    }
  });
  html.$on(
    "change",
    /*change_handler*/
    ctx[13]
  );
  html.$on(
    "click",
    /*click_handler*/
    ctx[14]
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      t0 = space();
      create_component(statustracker.$$.fragment);
      t1 = space();
      div = element("div");
      create_component(html.$$.fragment);
      this.h();
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      t0 = claim_space(nodes);
      claim_component(statustracker.$$.fragment, nodes);
      t1 = claim_space(nodes);
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      claim_component(html.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      var _a, _b;
      attr(div, "class", "html-container svelte-phx28p");
      toggle_class(
        div,
        "padding",
        /*padding*/
        ctx[11]
      );
      toggle_class(
        div,
        "pending",
        /*loading_status*/
        ((_a = ctx[5]) == null ? void 0 : _a.status) === "pending"
      );
      set_style(
        div,
        "min-height",
        /*min_height*/
        ctx[8] && /*loading_status*/
        ((_b = ctx[5]) == null ? void 0 : _b.status) !== "pending" ? css_units(
          /*min_height*/
          ctx[8]
        ) : void 0
      );
      set_style(
        div,
        "max-height",
        /*max_height*/
        ctx[9] ? css_units(
          /*max_height*/
          ctx[9]
        ) : void 0
      );
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t0, anchor);
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t1, anchor);
      insert_hydration(target, div, anchor);
      mount_component(html, div, null);
      current = true;
    },
    p(ctx2, dirty) {
      var _a, _b;
      if (
        /*show_label*/
        ctx2[7]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*show_label*/
          128) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(t0.parentNode, t0);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      const statustracker_changes = dirty & /*gradio, loading_status*/
      96 ? get_spread_update(statustracker_spread_levels, [
        dirty & /*gradio*/
        64 && { autoscroll: (
          /*gradio*/
          ctx2[6].autoscroll
        ) },
        dirty & /*gradio*/
        64 && { i18n: (
          /*gradio*/
          ctx2[6].i18n
        ) },
        dirty & /*loading_status*/
        32 && get_spread_object(
          /*loading_status*/
          ctx2[5]
        ),
        statustracker_spread_levels[3]
      ]) : {};
      statustracker.$set(statustracker_changes);
      const html_changes = {};
      if (dirty & /*value*/
      16)
        html_changes.value = /*value*/
        ctx2[4];
      if (dirty & /*elem_classes*/
      4)
        html_changes.elem_classes = /*elem_classes*/
        ctx2[2];
      if (dirty & /*visible*/
      8)
        html_changes.visible = /*visible*/
        ctx2[3];
      html.$set(html_changes);
      if (!current || dirty & /*padding*/
      2048) {
        toggle_class(
          div,
          "padding",
          /*padding*/
          ctx2[11]
        );
      }
      if (!current || dirty & /*loading_status*/
      32) {
        toggle_class(
          div,
          "pending",
          /*loading_status*/
          ((_a = ctx2[5]) == null ? void 0 : _a.status) === "pending"
        );
      }
      if (dirty & /*min_height, loading_status*/
      288) {
        set_style(
          div,
          "min-height",
          /*min_height*/
          ctx2[8] && /*loading_status*/
          ((_b = ctx2[5]) == null ? void 0 : _b.status) !== "pending" ? css_units(
            /*min_height*/
            ctx2[8]
          ) : void 0
        );
      }
      if (dirty & /*max_height*/
      512) {
        set_style(
          div,
          "max-height",
          /*max_height*/
          ctx2[9] ? css_units(
            /*max_height*/
            ctx2[9]
          ) : void 0
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(statustracker.$$.fragment, local);
      transition_in(html.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(statustracker.$$.fragment, local);
      transition_out(html.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(t1);
        detach(div);
      }
      if (if_block)
        if_block.d(detaching);
      destroy_component(statustracker, detaching);
      destroy_component(html);
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
        ctx[10]
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
      1024)
        block_changes.container = /*container*/
        ctx2[10];
      if (dirty & /*$$scope, padding, loading_status, min_height, max_height, value, elem_classes, visible, gradio, show_label, label*/
      35837) {
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
  let { label = "HTML" } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = "" } = $$props;
  let { loading_status } = $$props;
  let { gradio } = $$props;
  let { show_label = false } = $$props;
  let { min_height = void 0 } = $$props;
  let { max_height = void 0 } = $$props;
  let { container = false } = $$props;
  let { padding = true } = $$props;
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const change_handler = () => gradio.dispatch("change");
  const click_handler = () => gradio.dispatch("click");
  $$self.$$set = ($$props2) => {
    if ("label" in $$props2)
      $$invalidate(0, label = $$props2.label);
    if ("elem_id" in $$props2)
      $$invalidate(1, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(2, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(3, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(4, value = $$props2.value);
    if ("loading_status" in $$props2)
      $$invalidate(5, loading_status = $$props2.loading_status);
    if ("gradio" in $$props2)
      $$invalidate(6, gradio = $$props2.gradio);
    if ("show_label" in $$props2)
      $$invalidate(7, show_label = $$props2.show_label);
    if ("min_height" in $$props2)
      $$invalidate(8, min_height = $$props2.min_height);
    if ("max_height" in $$props2)
      $$invalidate(9, max_height = $$props2.max_height);
    if ("container" in $$props2)
      $$invalidate(10, container = $$props2.container);
    if ("padding" in $$props2)
      $$invalidate(11, padding = $$props2.padding);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*label, gradio*/
    65) {
      gradio.dispatch("change");
    }
  };
  return [
    label,
    elem_id,
    elem_classes,
    visible,
    value,
    loading_status,
    gradio,
    show_label,
    min_height,
    max_height,
    container,
    padding,
    clear_status_handler,
    change_handler,
    click_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      label: 0,
      elem_id: 1,
      elem_classes: 2,
      visible: 3,
      value: 4,
      loading_status: 5,
      gradio: 6,
      show_label: 7,
      min_height: 8,
      max_height: 9,
      container: 10,
      padding: 11
    });
  }
}
export {
  Index as default
};
