import { writable } from "svelte/store";
/**
 * Creates a command node
 * @param command command to add to the node
 * @returns a command node
 */
function command_node(command) {
    return {
        command: command || null,
        next: null,
        previous: null,
        push: function (command) {
            const node = command_node(command);
            node.previous = this;
            this.next = node;
        }
    };
}
/**
 * Creates a command manager
 * @returns a command manager
 */
export function command_manager() {
    let history = command_node();
    const can_undo = writable(false);
    const can_redo = writable(false);
    const current_history = writable(history);
    return {
        undo: function () {
            if (history.previous) {
                history.command?.undo();
                history = history.previous;
            }
            can_undo.set(!!history.previous);
            can_redo.set(!!history.next);
            current_history.set(history);
        },
        redo: function () {
            if (history.next) {
                history.next.command?.execute();
                history = history.next;
            }
            can_undo.set(!!history.previous);
            can_redo.set(!!history.next);
            current_history.set(history);
        },
        execute: function (command) {
            command.execute();
            history.push(command);
            history = history.next;
            can_undo.set(!!history.previous);
            can_redo.set(!!history.next);
            current_history.set(history);
        },
        hydrate: function (full_history) {
            setTimeout(() => {
                while (full_history.next) {
                    this.execute(full_history.next.command);
                    full_history = full_history.next;
                }
            }, 1000);
        },
        can_undo,
        can_redo,
        current_history,
        reset: function () {
            history = command_node();
            can_undo.set(false);
            can_redo.set(false);
            current_history.set(history);
        }
    };
}
