import { SvelteComponent, init, safe_not_equal, create_slot, element, space, claim_element, children, get_svelte_dataset, claim_space, detach, attr, set_style, toggle_class, insert_hydration, append_hydration, listen, update_slot_base, get_all_dirty_from_scope, get_slot_changes, transition_in, transition_out, createEventDispatcher, onMount, binding_callbacks, assign, bind, create_component, claim_component, mount_component, get_spread_update, get_spread_object, add_flush_callback, destroy_component } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { S as Static } from "./2.8WKXZUMv.js";
function create_fragment$1(ctx) {
  let div2;
  let button;
  let textContent = `<div class="chevron svelte-1qwdacm"><span class="chevron-left svelte-1qwdacm"></span></div>`;
  let t;
  let div1;
  let current;
  let mounted;
  let dispose;
  const default_slot_template = (
    /*#slots*/
    ctx[7].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[6],
    null
  );
  return {
    c() {
      div2 = element("div");
      button = element("button");
      button.innerHTML = textContent;
      t = space();
      div1 = element("div");
      if (default_slot)
        default_slot.c();
      this.h();
    },
    l(nodes) {
      div2 = claim_element(nodes, "DIV", { class: true, style: true });
      var div2_nodes = children(div2);
      button = claim_element(div2_nodes, "BUTTON", {
        class: true,
        "aria-label": true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button) !== "svelte-k78zcg")
        button.innerHTML = textContent;
      t = claim_space(div2_nodes);
      div1 = claim_element(div2_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      if (default_slot)
        default_slot.l(div1_nodes);
      div1_nodes.forEach(detach);
      div2_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "toggle-button svelte-1qwdacm");
      attr(button, "aria-label", "Toggle Sidebar");
      attr(div1, "class", "sidebar-content svelte-1qwdacm");
      attr(div2, "class", "sidebar svelte-1qwdacm");
      set_style(
        div2,
        "width",
        /*width_css*/
        ctx[3]
      );
      set_style(div2, "left", "calc(" + /*width_css*/
      ctx[3] + " * -1)");
      toggle_class(
        div2,
        "open",
        /*_open*/
        ctx[0]
      );
    },
    m(target, anchor) {
      insert_hydration(target, div2, anchor);
      append_hydration(div2, button);
      append_hydration(div2, t);
      append_hydration(div2, div1);
      if (default_slot) {
        default_slot.m(div1, null);
      }
      ctx[9](div2);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*click_handler*/
          ctx[8]
        );
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (default_slot) {
        if (default_slot.p && (!current || dirty & /*$$scope*/
        64)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[6],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[6]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[6],
              dirty,
              null
            ),
            null
          );
        }
      }
      if (!current || dirty & /*_open*/
      1) {
        toggle_class(
          div2,
          "open",
          /*_open*/
          ctx2[0]
        );
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
      if (detaching) {
        detach(div2);
      }
      if (default_slot)
        default_slot.d(detaching);
      ctx[9](null);
      mounted = false;
      dispose();
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  const dispatch = createEventDispatcher();
  let { open = true } = $$props;
  let { width } = $$props;
  let _open = false;
  let sidebar_div;
  let overlap_amount = 0;
  let width_css = typeof width === "number" ? `${width}px` : width;
  function check_overlap() {
    if (!(sidebar_div == null ? void 0 : sidebar_div.parentElement))
      return;
    const parent_rect = sidebar_div.parentElement.getBoundingClientRect();
    const sidebar_rect = sidebar_div.getBoundingClientRect();
    const available_space = parent_rect.left;
    overlap_amount = Math.max(0, sidebar_rect.width - available_space + 30);
  }
  onMount(() => {
    var _a;
    (_a = sidebar_div.parentElement) == null ? void 0 : _a.classList.add("sidebar-parent");
    check_overlap();
    window.addEventListener("resize", check_overlap);
    const update_parent_overlap = () => {
      if (sidebar_div == null ? void 0 : sidebar_div.parentElement) {
        sidebar_div.parentElement.style.setProperty("--overlap-amount", `${overlap_amount}px`);
      }
    };
    update_parent_overlap();
    $$invalidate(0, _open = open);
    return () => window.removeEventListener("resize", check_overlap);
  });
  const click_handler = () => {
    $$invalidate(0, _open = !_open);
    if (_open) {
      dispatch("expand");
    } else {
      dispatch("collapse");
    }
  };
  function div2_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      sidebar_div = $$value;
      $$invalidate(1, sidebar_div);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("open" in $$props2)
      $$invalidate(4, open = $$props2.open);
    if ("width" in $$props2)
      $$invalidate(5, width = $$props2.width);
    if ("$$scope" in $$props2)
      $$invalidate(6, $$scope = $$props2.$$scope);
  };
  return [
    _open,
    sidebar_div,
    dispatch,
    width_css,
    open,
    width,
    $$scope,
    slots,
    click_handler,
    div2_binding
  ];
}
class Sidebar extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance$1, create_fragment$1, safe_not_equal, { open: 4, width: 5 });
  }
}
function create_default_slot(ctx) {
  let current;
  const default_slot_template = (
    /*#slots*/
    ctx[4].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[8],
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
        256)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[8],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[8]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[8],
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
  let statustracker;
  let t;
  let sidebar;
  let updating_open;
  let current;
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[2].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[2].i18n
    ) },
    /*loading_status*/
    ctx[1]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  function sidebar_open_binding(value) {
    ctx[5](value);
  }
  let sidebar_props = {
    width: (
      /*width*/
      ctx[3]
    ),
    $$slots: { default: [create_default_slot] },
    $$scope: { ctx }
  };
  if (
    /*open*/
    ctx[0] !== void 0
  ) {
    sidebar_props.open = /*open*/
    ctx[0];
  }
  sidebar = new Sidebar({ props: sidebar_props });
  binding_callbacks.push(() => bind(sidebar, "open", sidebar_open_binding));
  sidebar.$on(
    "expand",
    /*expand_handler*/
    ctx[6]
  );
  sidebar.$on(
    "collapse",
    /*collapse_handler*/
    ctx[7]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      create_component(sidebar.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(sidebar.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(sidebar, target, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const statustracker_changes = dirty & /*gradio, loading_status*/
      6 ? get_spread_update(statustracker_spread_levels, [
        dirty & /*gradio*/
        4 && { autoscroll: (
          /*gradio*/
          ctx2[2].autoscroll
        ) },
        dirty & /*gradio*/
        4 && { i18n: (
          /*gradio*/
          ctx2[2].i18n
        ) },
        dirty & /*loading_status*/
        2 && get_spread_object(
          /*loading_status*/
          ctx2[1]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const sidebar_changes = {};
      if (dirty & /*width*/
      8)
        sidebar_changes.width = /*width*/
        ctx2[3];
      if (dirty & /*$$scope*/
      256) {
        sidebar_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_open && dirty & /*open*/
      1) {
        updating_open = true;
        sidebar_changes.open = /*open*/
        ctx2[0];
        add_flush_callback(() => updating_open = false);
      }
      sidebar.$set(sidebar_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(sidebar.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(sidebar.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(statustracker, detaching);
      destroy_component(sidebar, detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  let { open = true } = $$props;
  let { loading_status } = $$props;
  let { gradio } = $$props;
  let { width } = $$props;
  function sidebar_open_binding(value) {
    open = value;
    $$invalidate(0, open);
  }
  const expand_handler = () => gradio.dispatch("expand");
  const collapse_handler = () => gradio.dispatch("collapse");
  $$self.$$set = ($$props2) => {
    if ("open" in $$props2)
      $$invalidate(0, open = $$props2.open);
    if ("loading_status" in $$props2)
      $$invalidate(1, loading_status = $$props2.loading_status);
    if ("gradio" in $$props2)
      $$invalidate(2, gradio = $$props2.gradio);
    if ("width" in $$props2)
      $$invalidate(3, width = $$props2.width);
    if ("$$scope" in $$props2)
      $$invalidate(8, $$scope = $$props2.$$scope);
  };
  return [
    open,
    loading_status,
    gradio,
    width,
    slots,
    sidebar_open_binding,
    expand_handler,
    collapse_handler,
    $$scope
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      open: 0,
      loading_status: 1,
      gradio: 2,
      width: 3
    });
  }
}
export {
  Index as default
};
