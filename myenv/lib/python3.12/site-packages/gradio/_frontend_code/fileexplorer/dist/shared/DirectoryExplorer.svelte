<script>import FileTree from "./FileTree.svelte";
export let interactive;
export let file_count = "multiple";
export let value = [];
export let ls_fn;
let selected_folders = [];
const paths_equal = (path, path_2) => {
  return path.join("/") === path_2.join("/");
};
const path_in_set = (path, set) => {
  return set.some((x) => paths_equal(x, path));
};
const path_inside = (path, path_2) => {
  return path.join("/").startsWith(path_2.join("/"));
};
</script>

<div class="file-wrap">
	<FileTree
		path={[]}
		selected_files={value}
		{selected_folders}
		{interactive}
		{ls_fn}
		{file_count}
		valid_for_selection={false}
		on:check={(e) => {
			const { path, checked, type } = e.detail;
			if (checked) {
				if (file_count === "single") {
					value = [path];
				} else if (type === "folder") {
					if (!path_in_set(path, selected_folders)) {
						selected_folders = [...selected_folders, path];
					}
				} else {
					if (!path_in_set(path, value)) {
						value = [...value, path];
					}
				}
			} else {
				selected_folders = selected_folders.filter(
					(folder) => !path_inside(path, folder)
				); // deselect all parent folders
				if (type === "folder") {
					selected_folders = selected_folders.filter(
						(folder) => !path_inside(folder, path)
					); // deselect all children folders
					value = value.filter((file) => !path_inside(file, path)); // deselect all children files
				} else {
					value = value.filter((x) => !paths_equal(x, path));
				}
			}
		}}
	/>
</div>

<style>
	.file-wrap {
		height: calc(100% - 25px);
		overflow-y: scroll;
	}
</style>
