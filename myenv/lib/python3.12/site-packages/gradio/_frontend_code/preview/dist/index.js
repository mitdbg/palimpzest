import { spawn, spawnSync } from "node:child_process";
import * as net from "net";
import { join, dirname } from "path";
import * as fs from "fs";
import { createLogger, createServer, build } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import preprocess from "svelte-preprocess";
import { fileURLToPath } from "url";
const svelte_codes_to_ignore = {
  "reactive-component": "Icon"
};
const RE_SVELTE_IMPORT = /import\s+([\w*{},\s]+)\s+from\s+['"](svelte|svelte\/internal)['"]/g;
const RE_BARE_SVELTE_IMPORT = /import ("|')svelte(\/\w+)*("|')(;)*/g;
function plugins(config) {
  var _a, _b;
  const _additional_plugins = config.plugins || [];
  const _additional_svelte_preprocess = ((_a = config.svelte) == null ? void 0 : _a.preprocess) || [];
  const _svelte_extensions = (((_b = config.svelte) == null ? void 0 : _b.extensions) || [".svelte"]).map(
    (ext) => {
      if (ext.trim().startsWith(".")) {
        return ext;
      }
      return `.${ext.trim()}`;
    }
  );
  if (!_svelte_extensions.includes(".svelte")) {
    _svelte_extensions.push(".svelte");
  }
  return [
    svelte({
      inspector: false,
      onwarn(warning, handler) {
        if (svelte_codes_to_ignore.hasOwnProperty(warning.code) && svelte_codes_to_ignore[warning.code] && warning.message.includes(svelte_codes_to_ignore[warning.code])) {
          return;
        }
        handler(warning);
      },
      prebundleSvelteLibraries: false,
      hot: true,
      compilerOptions: {
        discloseVersion: false,
        hydratable: true
      },
      extensions: _svelte_extensions,
      preprocess: [
        preprocess({
          typescript: {
            compilerOptions: {
              declaration: false,
              declarationMap: false
            }
          }
        }),
        ..._additional_svelte_preprocess
      ]
    }),
    ..._additional_plugins
  ];
}
function make_gradio_plugin({
  mode,
  svelte_dir,
  backend_port,
  imports
}) {
  const v_id = "virtual:component-loader";
  const resolved_v_id = "\0" + v_id;
  return {
    name: "gradio",
    enforce: "pre",
    transform(code) {
      const new_code = code.replace(RE_SVELTE_IMPORT, (str, $1, $2) => {
        if ($1.trim().startsWith("type"))
          return str;
        const identifier = $1.trim().startsWith("* as") ? $1.replace("* as", "").trim() : $1.trim();
        return `const ${identifier.replace(
          " as ",
          ": "
        )} = window.__gradio__svelte__internal;`;
      }).replace(RE_BARE_SVELTE_IMPORT, "");
      return {
        code: new_code,
        map: null
      };
    },
    resolveId(id) {
      if (id === v_id) {
        return resolved_v_id;
      }
      if (id !== "svelte" && id !== "svelte/internal" && id.startsWith("svelte/")) {
        return join(svelte_dir, "svelte-submodules.js");
      }
    },
    load(id) {
      if (id === resolved_v_id) {
        return `export default {};`;
      }
    },
    transformIndexHtml(html) {
      return mode === "dev" ? [
        {
          tag: "script",
          children: `window.__GRADIO_DEV__ = "dev";
        window.__GRADIO__SERVER_PORT__ = ${backend_port};
        window.__GRADIO__CC__ = ${imports};`
        }
      ] : void 0;
    }
  };
}
const vite_messages_to_ignore = [
  "Default and named imports from CSS files are deprecated.",
  "The above dynamic import cannot be analyzed by Vite."
];
const logger = createLogger();
const originalWarning = logger.warn;
logger.warn = (msg, options) => {
  if (vite_messages_to_ignore.some((m) => msg.includes(m)))
    return;
  originalWarning(msg, options);
};
async function create_server({
  component_dir,
  root_dir,
  frontend_port,
  backend_port,
  host,
  python_path
}) {
  var _a;
  process.env.gradio_mode = "dev";
  const [imports, config] = await generate_imports(
    component_dir,
    root_dir,
    python_path
  );
  const svelte_dir = join(root_dir, "assets", "svelte");
  try {
    const server = await createServer({
      customLogger: logger,
      mode: "development",
      configFile: false,
      root: root_dir,
      server: {
        port: frontend_port,
        host,
        fs: {
          allow: [root_dir, component_dir]
        }
      },
      resolve: {
        conditions: ["gradio"]
      },
      build: {
        target: config.build.target
      },
      optimizeDeps: config.optimizeDeps,
      plugins: [
        ...plugins(config),
        make_gradio_plugin({
          mode: "dev",
          backend_port,
          svelte_dir,
          imports
        })
      ]
    });
    await server.listen();
    console.info(
      `[orange3]Frontend Server[/] (Go here): ${(_a = server.resolvedUrls) == null ? void 0 : _a.local}`
    );
  } catch (e) {
    console.error(e);
  }
}
function find_frontend_folders(start_path) {
  if (!fs.existsSync(start_path)) {
    console.warn("No directory found at:", start_path);
    return [];
  }
  if (fs.existsSync(join(start_path, "pyproject.toml")))
    return [start_path];
  const results = [];
  const dir = fs.readdirSync(start_path);
  dir.forEach((dir2) => {
    const filepath = join(start_path, dir2);
    if (fs.existsSync(filepath)) {
      if (fs.existsSync(join(filepath, "pyproject.toml")))
        results.push(filepath);
    }
  });
  return results;
}
function to_posix(_path) {
  const isExtendedLengthPath = /^\\\\\?\\/.test(_path);
  const hasNonAscii = /[^\u0000-\u0080]+/.test(_path);
  if (isExtendedLengthPath || hasNonAscii) {
    return _path;
  }
  return _path.replace(/\\/g, "/");
}
async function generate_imports(component_dir, root, python_path) {
  const components = find_frontend_folders(component_dir);
  const component_entries = components.flatMap((component) => {
    return examine_module(component, root, python_path, "dev");
  });
  if (component_entries.length === 0) {
    console.info(
      `No custom components were found in ${component_dir}. It is likely that dev mode does not work properly. Please pass the --gradio-path and --python-path CLI arguments so that gradio uses the right executables.`
    );
  }
  let component_config = {
    plugins: [],
    svelte: {
      preprocess: []
    },
    build: {
      target: []
    },
    optimizeDeps: {}
  };
  await Promise.all(
    component_entries.map(async (component) => {
      var _a, _b;
      if (component.frontend_dir && fs.existsSync(join(component.frontend_dir, "gradio.config.js"))) {
        const m = await import(join("file://" + component.frontend_dir, "gradio.config.js"));
        component_config.plugins = m.default.plugins || [];
        component_config.svelte.preprocess = ((_a = m.default.svelte) == null ? void 0 : _a.preprocess) || [];
        component_config.build.target = ((_b = m.default.build) == null ? void 0 : _b.target) || "modules";
        component_config.optimizeDeps = m.default.optimizeDeps || {};
      }
    })
  );
  const imports = component_entries.reduce((acc, component) => {
    const pkg = JSON.parse(
      fs.readFileSync(join(component.frontend_dir, "package.json"), "utf-8")
    );
    const exports = {
      component: pkg.exports["."],
      example: pkg.exports["./example"]
    };
    if (!exports.component)
      throw new Error(
        "Could not find component entry point. Please check the exports field of your package.json."
      );
    const example = exports.example ? `example: () => import("/@fs/${to_posix(
      join(component.frontend_dir, exports.example.gradio)
    )}"),
` : "";
    return `${acc}"${component.component_class_id}": {
			${example}
			component: () => import("/@fs/${to_posix(
      join(component.frontend_dir, exports.component.gradio)
    )}")
			},
`;
  }, "");
  return [`{${imports}}`, component_config];
}
async function make_build({
  component_dir,
  root_dir,
  python_path
}) {
  var _a, _b;
  process.env.gradio_mode = "dev";
  const svelte_dir = join(root_dir, "assets", "svelte");
  const module_meta = examine_module(
    component_dir,
    root_dir,
    python_path,
    "build"
  );
  try {
    for (const comp of module_meta) {
      const template_dir = comp.template_dir;
      const source_dir = comp.frontend_dir;
      const pkg = JSON.parse(
        fs.readFileSync(join(source_dir, "package.json"), "utf-8")
      );
      let component_config = {
        plugins: [],
        svelte: {
          preprocess: []
        },
        build: {
          target: []
        },
        optimizeDeps: {}
      };
      if (comp.frontend_dir && fs.existsSync(join(comp.frontend_dir, "gradio.config.js"))) {
        const m = await import(join("file://" + comp.frontend_dir, "gradio.config.js"));
        component_config.plugins = m.default.plugins || [];
        component_config.svelte.preprocess = ((_a = m.default.svelte) == null ? void 0 : _a.preprocess) || [];
        component_config.build.target = ((_b = m.default.build) == null ? void 0 : _b.target) || "modules";
        component_config.optimizeDeps = m.default.optimizeDeps || {};
      }
      const exports = [
        ["component", pkg.exports["."]],
        ["example", pkg.exports["./example"]]
      ].filter(([_, path]) => !!path);
      for (const [entry, path] of exports) {
        try {
          const x = await build({
            root: source_dir,
            configFile: false,
            plugins: [
              ...plugins(component_config),
              make_gradio_plugin({ mode: "build", svelte_dir })
            ],
            resolve: {
              conditions: ["gradio"]
            },
            build: {
              target: component_config.build.target,
              emptyOutDir: true,
              outDir: join(template_dir, entry),
              lib: {
                entry: join(source_dir, path.gradio),
                fileName: "index.js",
                formats: ["es"]
              },
              minify: true,
              rollupOptions: {
                output: {
                  entryFileNames: (chunkInfo) => {
                    if (chunkInfo.isEntry) {
                      return "index.js";
                    }
                    return `${chunkInfo.name.toLocaleLowerCase()}.js`;
                  }
                }
              }
            }
          });
        } catch (e) {
          throw e;
        }
      }
    }
  } catch (e) {
    throw e;
  }
}
const __dirname = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
function parse_args(args2) {
  const arg_map = {};
  for (let i = 0; i < args2.length; i++) {
    const arg = args2[i];
    if (arg.startsWith("--")) {
      const name = arg.slice(2);
      const value = args2[i + 1];
      arg_map[name] = value;
      i++;
    }
  }
  return arg_map;
}
const parsed_args = parse_args(args);
async function run() {
  if (parsed_args.mode === "build") {
    await make_build({
      component_dir: parsed_args["component-directory"],
      root_dir: parsed_args.root,
      python_path: parsed_args["python-path"]
    });
  } else {
    let std_out = function(mode) {
      return function(data) {
        const _data = data.toString();
        if (_data.includes("Running on")) {
          create_server({
            component_dir: options.component_dir,
            root_dir: options.root_dir,
            frontend_port,
            backend_port,
            host: options.host,
            python_path: parsed_args["python-path"]
          });
        }
        process[mode].write(_data);
      };
    };
    const [backend_port, frontend_port] = await find_free_ports(7860, 8860);
    const options = {
      component_dir: parsed_args["component-directory"],
      root_dir: parsed_args.root,
      frontend_port,
      backend_port,
      host: parsed_args.host,
      ...parsed_args
    };
    process.env.GRADIO_BACKEND_PORT = backend_port.toString();
    const _process = spawn(
      parsed_args["gradio-path"],
      [parsed_args.app, "--watch-dirs", options.component_dir],
      {
        shell: true,
        stdio: "pipe",
        cwd: process.cwd(),
        env: {
          ...process.env,
          GRADIO_SERVER_PORT: backend_port.toString(),
          PYTHONUNBUFFERED: "true"
        }
      }
    );
    _process.stdout.setEncoding("utf8");
    _process.stderr.setEncoding("utf8");
    _process.stdout.on("data", std_out("stdout"));
    _process.stderr.on("data", std_out("stderr"));
    _process.on("exit", () => kill_process(_process));
    _process.on("close", () => kill_process(_process));
    _process.on("disconnect", () => kill_process(_process));
  }
}
function kill_process(process2) {
  process2.kill("SIGKILL");
}
run();
async function find_free_ports(start_port, end_port) {
  let found_ports = [];
  for (let port = start_port; port < end_port; port++) {
    if (await is_free_port(port)) {
      found_ports.push(port);
      if (found_ports.length === 2) {
        return [found_ports[0], found_ports[1]];
      }
    }
  }
  throw new Error(
    `Could not find free ports: there were not enough ports available.`
  );
}
function is_free_port(port) {
  return new Promise((accept, reject) => {
    const sock = net.createConnection(port, "127.0.0.1");
    setTimeout(() => {
      sock.destroy();
      reject(
        new Error(`Timeout while detecting free port with 127.0.0.1:${port} `)
      );
    }, 3e3);
    sock.once("connect", () => {
      sock.end();
      accept(false);
    });
    sock.once("error", (e) => {
      sock.destroy();
      if (e.code === "ECONNREFUSED") {
        accept(true);
      } else {
        reject(e);
      }
    });
  });
}
function is_truthy(value) {
  return value !== null && value !== void 0 && value !== false;
}
function examine_module(component_dir, root, python_path, mode) {
  const _process = spawnSync(
    python_path,
    [join(__dirname, "examine.py"), "-m", mode],
    {
      cwd: join(component_dir, "backend"),
      stdio: "pipe"
    }
  );
  const exceptions = [];
  const components = _process.stdout.toString().trim().split("\n").map((line) => {
    if (line.startsWith("|EXCEPTION|")) {
      exceptions.push(line.slice("|EXCEPTION|:".length));
    }
    const [name, template_dir, frontend_dir, component_class_id] = line.split("~|~|~|~");
    if (name && template_dir && frontend_dir && component_class_id) {
      return {
        name: name.trim(),
        template_dir: template_dir.trim(),
        frontend_dir: frontend_dir.trim(),
        component_class_id: component_class_id.trim()
      };
    }
    return false;
  }).filter(is_truthy);
  if (exceptions.length > 0) {
    console.info(
      `While searching for gradio custom component source directories in ${component_dir}, the following exceptions were raised. If dev mode does not work properly please pass the --gradio-path and --python-path CLI arguments so that gradio uses the right executables: ${exceptions.join(
        "\n"
      )}`
    );
  }
  return components;
}
export {
  create_server,
  examine_module,
  find_free_ports,
  is_free_port
};
