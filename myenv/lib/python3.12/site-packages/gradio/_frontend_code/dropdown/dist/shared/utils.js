function positive_mod(n, m) {
    return ((n % m) + m) % m;
}
export function handle_filter(choices, input_text) {
    return choices.reduce((filtered_indices, o, index) => {
        if (input_text ? o[0].toLowerCase().includes(input_text.toLowerCase()) : true) {
            filtered_indices.push(index);
        }
        return filtered_indices;
    }, []);
}
export function handle_change(dispatch, value, value_is_output) {
    dispatch("change", value);
    if (!value_is_output) {
        dispatch("input");
    }
}
export function handle_shared_keys(e, active_index, filtered_indices) {
    if (e.key === "Escape") {
        return [false, active_index];
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (filtered_indices.length >= 0) {
            if (active_index === null) {
                active_index =
                    e.key === "ArrowDown"
                        ? filtered_indices[0]
                        : filtered_indices[filtered_indices.length - 1];
            }
            else {
                const index_in_filtered = filtered_indices.indexOf(active_index);
                const increment = e.key === "ArrowUp" ? -1 : 1;
                active_index =
                    filtered_indices[positive_mod(index_in_filtered + increment, filtered_indices.length)];
            }
        }
    }
    return [true, active_index];
}
