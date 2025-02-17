/**
 * Svelte action to handle clicks outside of a DOM node
 * @param node DOM node to check the click is outside of
 * @param callback callback function to call if click is outside
 * @returns svelte action return object with destroy method to remove event listener
 */
export declare function click_outside(node: Node, callback: (arg: MouseEvent) => void): any;
