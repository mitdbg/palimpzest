import { SvelteComponent, init, safe_not_equal, element, create_component, claim_element, children, claim_component, detach, attr, toggle_class, set_style, insert_hydration, mount_component, transition_in, transition_out, destroy_component, createEventDispatcher, getContext, component_subscribe, onMount, tick, create_slot, update_slot_base, get_all_dirty_from_scope, get_slot_changes } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { a as TABS } from "./Tabs.Bw409lYA.js";
import { k as Index$1 } from "./2.8WKXZUMv.js";
function create_default_slot$1(ctx) {
  let current;
  const default_slot_template = (
    /*#slots*/
    ctx[13].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[14],
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
        16384)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[14],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[14]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[14],
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
function create_fragment$1(ctx) {
  let div;
  let column;
  let div_class_value;
  let current;
  column = new Index$1({
    props: {
      scale: (
        /*scale*/
        ctx[4] >= 1 ? (
          /*scale*/
          ctx[4]
        ) : null
      ),
      $$slots: { default: [create_default_slot$1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      div = element("div");
      create_component(column.$$.fragment);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { id: true, class: true, role: true });
      var div_nodes = children(div);
      claim_component(column.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(
        div,
        "id",
        /*elem_id*/
        ctx[0]
      );
      attr(div, "class", div_class_value = "tabitem " + /*elem_classes*/
      ctx[1].join(" ") + " svelte-wv8on1");
      attr(div, "role", "tabpanel");
      toggle_class(
        div,
        "grow-children",
        /*scale*/
        ctx[4] >= 1
      );
      set_style(
        div,
        "display",
        /*$selected_tab*/
        ctx[5] === /*id*/
        ctx[2] && /*visible*/
        ctx[3] ? "flex" : "none"
      );
      set_style(
        div,
        "flex-grow",
        /*scale*/
        ctx[4]
      );
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      mount_component(column, div, null);
      current = true;
    },
    p(ctx2, [dirty]) {
      const column_changes = {};
      if (dirty & /*scale*/
      16)
        column_changes.scale = /*scale*/
        ctx2[4] >= 1 ? (
          /*scale*/
          ctx2[4]
        ) : null;
      if (dirty & /*$$scope*/
      16384) {
        column_changes.$$scope = { dirty, ctx: ctx2 };
      }
      column.$set(column_changes);
      if (!current || dirty & /*elem_id*/
      1) {
        attr(
          div,
          "id",
          /*elem_id*/
          ctx2[0]
        );
      }
      if (!current || dirty & /*elem_classes*/
      2 && div_class_value !== (div_class_value = "tabitem " + /*elem_classes*/
      ctx2[1].join(" ") + " svelte-wv8on1")) {
        attr(div, "class", div_class_value);
      }
      if (!current || dirty & /*elem_classes, scale*/
      18) {
        toggle_class(
          div,
          "grow-children",
          /*scale*/
          ctx2[4] >= 1
        );
      }
      if (dirty & /*$selected_tab, id, visible*/
      44) {
        set_style(
          div,
          "display",
          /*$selected_tab*/
          ctx2[5] === /*id*/
          ctx2[2] && /*visible*/
          ctx2[3] ? "flex" : "none"
        );
      }
      if (dirty & /*scale*/
      16) {
        set_style(
          div,
          "flex-grow",
          /*scale*/
          ctx2[4]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(column.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(column.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_component(column);
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let $selected_tab_index;
  let $selected_tab;
  let { $$slots: slots = {}, $$scope } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { label } = $$props;
  let { id = {} } = $$props;
  let { visible } = $$props;
  let { interactive } = $$props;
  let { order } = $$props;
  let { scale } = $$props;
  const dispatch = createEventDispatcher();
  const { register_tab, unregister_tab, selected_tab, selected_tab_index } = getContext(TABS);
  component_subscribe($$self, selected_tab, (value) => $$invalidate(5, $selected_tab = value));
  component_subscribe($$self, selected_tab_index, (value) => $$invalidate(12, $selected_tab_index = value));
  let tab_index;
  onMount(() => {
    return () => unregister_tab({ label, id, elem_id }, order);
  });
  $$self.$$set = ($$props2) => {
    if ("elem_id" in $$props2)
      $$invalidate(0, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(1, elem_classes = $$props2.elem_classes);
    if ("label" in $$props2)
      $$invalidate(8, label = $$props2.label);
    if ("id" in $$props2)
      $$invalidate(2, id = $$props2.id);
    if ("visible" in $$props2)
      $$invalidate(3, visible = $$props2.visible);
    if ("interactive" in $$props2)
      $$invalidate(9, interactive = $$props2.interactive);
    if ("order" in $$props2)
      $$invalidate(10, order = $$props2.order);
    if ("scale" in $$props2)
      $$invalidate(4, scale = $$props2.scale);
    if ("$$scope" in $$props2)
      $$invalidate(14, $$scope = $$props2.$$scope);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*label, id, elem_id, visible, interactive, scale, order*/
    1821) {
      $$invalidate(11, tab_index = register_tab(
        {
          label,
          id,
          elem_id,
          visible,
          interactive,
          scale
        },
        order
      ));
    }
    if ($$self.$$.dirty & /*$selected_tab_index, tab_index, label*/
    6400) {
      $selected_tab_index === tab_index && tick().then(() => dispatch("select", { value: label, index: tab_index }));
    }
  };
  return [
    elem_id,
    elem_classes,
    id,
    visible,
    scale,
    $selected_tab,
    selected_tab,
    selected_tab_index,
    label,
    interactive,
    order,
    tab_index,
    $selected_tab_index,
    slots,
    $$scope
  ];
}
class TabItem extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$1, create_fragment$1, safe_not_equal, {
      elem_id: 0,
      elem_classes: 1,
      label: 8,
      id: 2,
      visible: 3,
      interactive: 9,
      order: 10,
      scale: 4
    });
  }
}
const TabItem$1 = TabItem;
function create_default_slot(ctx) {
  let current;
  const default_slot_template = (
    /*#slots*/
    ctx[9].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[11],
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
        2048)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[11],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[11]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[11],
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
function create_fragment(ctx) {
  let tabitem;
  let current;
  tabitem = new TabItem$1({
    props: {
      elem_id: (
        /*elem_id*/
        ctx[0]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[1]
      ),
      label: (
        /*label*/
        ctx[2]
      ),
      visible: (
        /*visible*/
        ctx[5]
      ),
      interactive: (
        /*interactive*/
        ctx[6]
      ),
      id: (
        /*id*/
        ctx[3]
      ),
      order: (
        /*order*/
        ctx[7]
      ),
      scale: (
        /*scale*/
        ctx[8]
      ),
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  tabitem.$on(
    "select",
    /*select_handler*/
    ctx[10]
  );
  return {
    c() {
      create_component(tabitem.$$.fragment);
    },
    l(nodes) {
      claim_component(tabitem.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(tabitem, target, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const tabitem_changes = {};
      if (dirty & /*elem_id*/
      1)
        tabitem_changes.elem_id = /*elem_id*/
        ctx2[0];
      if (dirty & /*elem_classes*/
      2)
        tabitem_changes.elem_classes = /*elem_classes*/
        ctx2[1];
      if (dirty & /*label*/
      4)
        tabitem_changes.label = /*label*/
        ctx2[2];
      if (dirty & /*visible*/
      32)
        tabitem_changes.visible = /*visible*/
        ctx2[5];
      if (dirty & /*interactive*/
      64)
        tabitem_changes.interactive = /*interactive*/
        ctx2[6];
      if (dirty & /*id*/
      8)
        tabitem_changes.id = /*id*/
        ctx2[3];
      if (dirty & /*order*/
      128)
        tabitem_changes.order = /*order*/
        ctx2[7];
      if (dirty & /*scale*/
      256)
        tabitem_changes.scale = /*scale*/
        ctx2[8];
      if (dirty & /*$$scope*/
      2048) {
        tabitem_changes.$$scope = { dirty, ctx: ctx2 };
      }
      tabitem.$set(tabitem_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(tabitem.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(tabitem.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(tabitem, detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { label } = $$props;
  let { id } = $$props;
  let { gradio } = $$props;
  let { visible = true } = $$props;
  let { interactive = true } = $$props;
  let { order } = $$props;
  let { scale } = $$props;
  const select_handler = ({ detail }) => gradio == null ? void 0 : gradio.dispatch("select", detail);
  $$self.$$set = ($$props2) => {
    if ("elem_id" in $$props2)
      $$invalidate(0, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(1, elem_classes = $$props2.elem_classes);
    if ("label" in $$props2)
      $$invalidate(2, label = $$props2.label);
    if ("id" in $$props2)
      $$invalidate(3, id = $$props2.id);
    if ("gradio" in $$props2)
      $$invalidate(4, gradio = $$props2.gradio);
    if ("visible" in $$props2)
      $$invalidate(5, visible = $$props2.visible);
    if ("interactive" in $$props2)
      $$invalidate(6, interactive = $$props2.interactive);
    if ("order" in $$props2)
      $$invalidate(7, order = $$props2.order);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("$$scope" in $$props2)
      $$invalidate(11, $$scope = $$props2.$$scope);
  };
  return [
    elem_id,
    elem_classes,
    label,
    id,
    gradio,
    visible,
    interactive,
    order,
    scale,
    slots,
    select_handler,
    $$scope
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      elem_id: 0,
      elem_classes: 1,
      label: 2,
      id: 3,
      gradio: 4,
      visible: 5,
      interactive: 6,
      order: 7,
      scale: 8
    });
  }
}
export {
  TabItem$1 as BaseTabItem,
  Index as default
};
