import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, space, empty, claim_space, transition_in, group_outros, transition_out, check_outros, createEventDispatcher, onMount, create_component, claim_component, mount_component, destroy_component } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { I as IconButton } from "./2.8WKXZUMv.js";
import { M as Maximize, a as Minimize } from "./Minimize.BNzzPy3I.js";
function create_fragment$1(ctx) {
  let svg;
  let rect;
  let circle;
  let polyline;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      circle = svg_element("circle");
      polyline = svg_element("polyline");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        xmlns: true,
        width: true,
        height: true,
        viewBox: true,
        fill: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        class: true
      });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        rx: true,
        ry: true
      });
      children(rect).forEach(detach);
      circle = claim_svg_element(svg_nodes, "circle", { cx: true, cy: true, r: true });
      children(circle).forEach(detach);
      polyline = claim_svg_element(svg_nodes, "polyline", { points: true });
      children(polyline).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "3");
      attr(rect, "y", "3");
      attr(rect, "width", "18");
      attr(rect, "height", "18");
      attr(rect, "rx", "2");
      attr(rect, "ry", "2");
      attr(circle, "cx", "8.5");
      attr(circle, "cy", "8.5");
      attr(circle, "r", "1.5");
      attr(polyline, "points", "21 15 16 10 5 21");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "width", "100%");
      attr(svg, "height", "100%");
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "fill", "none");
      attr(svg, "stroke", "currentColor");
      attr(svg, "stroke-width", "1.5");
      attr(svg, "stroke-linecap", "round");
      attr(svg, "stroke-linejoin", "round");
      attr(svg, "class", "feather feather-image");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, circle);
      append_hydration(svg, polyline);
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
class Image extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$1, safe_not_equal, {});
  }
}
function create_if_block_1(ctx) {
  let iconbutton;
  let current;
  iconbutton = new IconButton({
    props: {
      Icon: Maximize,
      label: "View in full screen"
    }
  });
  iconbutton.$on(
    "click",
    /*toggle_full_screen*/
    ctx[1]
  );
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
    p: noop,
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
function create_if_block(ctx) {
  let iconbutton;
  let current;
  iconbutton = new IconButton({
    props: {
      Icon: Minimize,
      label: "Exit full screen"
    }
  });
  iconbutton.$on(
    "click",
    /*toggle_full_screen*/
    ctx[1]
  );
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
    p: noop,
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
function create_fragment(ctx) {
  let t;
  let if_block1_anchor;
  let current;
  let if_block0 = !/*is_full_screen*/
  ctx[0] && create_if_block_1(ctx);
  let if_block1 = (
    /*is_full_screen*/
    ctx[0] && create_if_block(ctx)
  );
  return {
    c() {
      if (if_block0)
        if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      if_block1_anchor = empty();
    },
    l(nodes) {
      if (if_block0)
        if_block0.l(nodes);
      t = claim_space(nodes);
      if (if_block1)
        if_block1.l(nodes);
      if_block1_anchor = empty();
    },
    m(target, anchor) {
      if (if_block0)
        if_block0.m(target, anchor);
      insert_hydration(target, t, anchor);
      if (if_block1)
        if_block1.m(target, anchor);
      insert_hydration(target, if_block1_anchor, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      if (!/*is_full_screen*/
      ctx2[0]) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty & /*is_full_screen*/
          1) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_1(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(t.parentNode, t);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*is_full_screen*/
        ctx2[0]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty & /*is_full_screen*/
          1) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(if_block1_anchor.parentNode, if_block1_anchor);
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
        detach(t);
        detach(if_block1_anchor);
      }
      if (if_block0)
        if_block0.d(detaching);
      if (if_block1)
        if_block1.d(detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { container = void 0 } = $$props;
  const dispatch = createEventDispatcher();
  let is_full_screen = false;
  onMount(() => {
    document.addEventListener("fullscreenchange", () => {
      $$invalidate(0, is_full_screen = !!document.fullscreenElement);
      dispatch("fullscreenchange", is_full_screen);
    });
  });
  const toggle_full_screen = async () => {
    if (!container)
      return;
    if (!is_full_screen) {
      await container.requestFullscreen();
    } else {
      await document.exitFullscreen();
      $$invalidate(0, is_full_screen = !is_full_screen);
    }
  };
  $$self.$$set = ($$props2) => {
    if ("container" in $$props2)
      $$invalidate(2, container = $$props2.container);
  };
  return [is_full_screen, toggle_full_screen, container];
}
class FullscreenButton extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, { container: 2 });
  }
}
export {
  FullscreenButton as F,
  Image as I
};
