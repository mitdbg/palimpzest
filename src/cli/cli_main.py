import os
import subprocess
from typing import Tuple

import click
import yaml
from click_aliases import ClickAliasedGroup
from prettytable import PrettyTable


############ DEFINITIONS ############
class InvalidCommandError(Exception):
    pass


############## HELPERS ##############
def _print_msg(msg: str) -> None:
    """
    Helper function to print messages in CLI-specific format. Currently just a wrapper around print(),
    could easily be extended to include color/formatted output.

    Parameters
    ----------
    msg: str
        Message to print to the console.
    """
    print(f"{msg}")


def _run_bash_command(command: str) -> Tuple[str, str]:
    """
    Helper function to split a bash command on spaces and execute it using subprocess.

    Parameters
    ----------
    command: str
        Shell command to execute with subprocess.

    Returns
    -------
    Tuple[str, str]
        Tuple returning the stdout and stderr from running the shell command.
    """
    # split command on spaces into list of strings
    command_str_lst = command.split(" ")

    # execute command and capture the output
    out = subprocess.run(command_str_lst, capture_output=True)

    # return stdout as string
    return str(out.stdout, "utf-8"), str(out.stderr, "utf-8")


def _help() -> str:
    """
    Syntactic sugar to call `pz --help` when a user calls `pz help`.

    Returns
    -------
    str
        The help text for the pz CLI.
    """
    # execute the help command using subprocess and return output
    stdout, _ = _run_bash_command("pz --help")

    return stdout


############ CLICK API ##############
@click.group(cls=ClickAliasedGroup)
def cli():
    """
    The CLI tool for Palimpzest.
    """
    pass


@cli.command(aliases=["h"])
def help() -> None:
    """
    Print the help message for PZ.
    """
    _print_msg(_help())


@cli.command(aliases=["i"])
def init() -> None:
    """
    Initialize data directory for PZ.
    """
    # set directory and initialize it for PZ
    import palimpzest.datamanager.datamanager as pzdm
    from palimpzest.constants import PZ_DIR

    pzdm.DataDirectory()
    _print_msg(f"Palimpzest system initialized in: {PZ_DIR}")


@cli.command(aliases=["lsdata", "ls"])
def ls_data() -> None:
    """
    Print a table listing the datasets registered with PZ.
    """
    # fetch list of registered datasets
    # Time the loading of the library
    import palimpzest.datamanager.datamanager as pzdm

    ds = pzdm.DataDirectory().list_registered_datasets()

    # construct table for printing
    table = [["Name", "Type", "Path"]]
    for path, descriptor in ds:
        table.append([path, descriptor[0], descriptor[1]])

    # print table of registered datasets
    t = PrettyTable(table[0])
    t.add_rows(table[1:])
    _print_msg(str(t))
    _print_msg("")
    _print_msg(f"Total datasets: {len(table) - 1}")


@cli.command(aliases=["synthesize", "syn"])
@click.option("--name", type=str, default=None, required=True, help="Registered name for the set of synthetic values")
@click.option("--count", type=int, default=100, help="How many values should be synthesized")
def synthesize_data(name: str, count: int) -> None:
    """
    Register a synthetic set of values with PZ

    Parameters
    ----------
    name: str
        Name to register the data set with

    count: int
        The nunber of values to synthesize
    """
    import palimpzest.datamanager.datamanager as pzdm

    name = name.strip()

    vals = []
    for i in range(0, count):
        vals.append(i)
    pzdm.DataDirectory().register_dataset(vals, name)

    _print_msg(f"Registered {name}")


@cli.command(aliases=["register", "reg", "r"])
@click.option("--path", type=str, default=None, required=True, help="File or directory to register as dataset.")
@click.option("--name", type=str, default=None, required=True, help="Registered name for the file/dir.")
def register_data(path: str, name: str) -> None:
    """
    Register a data file or data directory with PZ.

    Parameters
    ----------
    path: str
        Path to the data file or directory to register with PZ.

    name: str
        Name to register the data file / directory with.
    """
    import palimpzest.datamanager.datamanager as pzdm

    # parse path and name
    path = path.strip()
    name = name.strip()

    # register dataset
    if os.path.isfile(path):
        pzdm.DataDirectory().register_local_file(os.path.abspath(path), name)

    elif os.path.isdir(path):
        pzdm.DataDirectory().register_local_directory(os.path.abspath(path), name)

    else:
        raise InvalidCommandError(f"Path {path} is invalid. Does not point to a file or directory.")

    _print_msg(f"Registered {name}")


@cli.command(aliases=["rmdata", "rm"])
@click.option("--name", type=str, default=None, required=True, help="Name of registered dataset to be removed.")
def rm_data(name: str) -> None:
    """
    Remove a dataset that was registered with PZ.

    Parameters
    ----------
    name: str
        Name of the dataset to unregister.
    """
    import palimpzest.datamanager.datamanager as pzdm

    # parse name
    name = name.strip()

    # remove dataset from registry
    pzdm.DataDirectory().rm_registered_dataset(name)

    _print_msg(f"Deleted {name}")


@cli.command(aliases=["clear", "clr"])
def clear_cache() -> None:
    """
    Clear the Palimpzest cache.
    """
    import palimpzest.datamanager.datamanager as pzdm

    pzdm.DataDirectory().clear_cache(keep_registry=True)
    _print_msg("Cache cleared")


@cli.command(aliases=["config", "pc"])
def print_config() -> None:
    """
    Print the current config that Palimpzest is using.
    """
    import palimpzest.datamanager.datamanager as pzdm

    # load config yaml file
    config = pzdm.DataDirectory().get_config()

    # print contents of config
    _print_msg(f"--- {config['name']} ---\n{yaml.dump(config)}")


@cli.command(aliases=["cc"])
@click.option("--name", type=str, default=None, required=True, help="Name of the config to create.")
@click.option(
    "--llmservice",
    type=click.Choice(["openai", "together", "google"], case_sensitive=False),
    default="openai",
    help="Name of the LLM service to use.",
)
@click.option("--parallel", type=bool, default=False, help="Whether to run operations in parallel or not.")
@click.option("--set", type=bool, is_flag=True, help="Set the created config to be the current config.")
def create_config(name: str, llmservice: str, parallel: bool, set: bool) -> None:
    """
    Create a Palimpzest config. You must set the `name` field. You may optionally
    set the `llmservice` and `parallel` fields (default to )

    Parameters
    ----------
    name: str
        Name of the config to create.
    llmservice: str
        Name of the LLM service to use.
    parallel: bool
        Whether to run operations in parallel or not.
    set: bool
        If this flag is present, it will set the created config to be
        the current config.
    """
    from palimpzest.config import Config
    from palimpzest.constants import PZ_DIR

    # check that config name is unique
    if os.path.exists(os.path.join(PZ_DIR, f"config_{name}.yaml")):
        raise InvalidCommandError(f"Config with name {name} already exists.")

    # create config
    config = Config(name, llmservice, parallel)

    # set newly created config to be the current config if specified
    if set:
        config.set_current_config()

    _print_msg(f"Created config: {name}" if set is False else f"Created and set config: {name}")


@cli.command(aliases=["rmconfig", "rmc"])
@click.option("--name", type=str, default=None, required=True, help="Name of the config to remove.")
def rm_config(name: str) -> None:
    """
    Remove the specified config from Palimpzest. You cannot remove the default config.
    If this config was the current config, the current config will be set to the default config.

    Parameters
    ----------
    name: str
        Name of the config to remove.
    """
    from palimpzest.config import Config
    from palimpzest.constants import PZ_DIR

    # check that config exists
    if not os.path.exists(os.path.join(PZ_DIR, f"config_{name}.yaml")):
        raise InvalidCommandError(f"Config with name {name} does not exist.")

    # load the specified config
    config = Config(name)

    # remove the config; this will update the current config as well
    config.remove_config()
    _print_msg(f"Deleted config: {name}")


@cli.command(aliases=["set", "sc"])
@click.option("--name", type=str, default=None, required=True, help="Name of the config to set as the current config.")
def set_config(name: str) -> None:
    """
    Set the current config for Palimpzest to use.

    Parameters
    ----------
    name: str
        Name of the config to set as the current config.
    """
    from palimpzest.config import Config
    from palimpzest.constants import PZ_DIR

    # check that config exists
    if not os.path.exists(os.path.join(PZ_DIR, f"config_{name}.yaml")):
        raise InvalidCommandError(f"Config with name {name} does not exist.")

    # load the specified config
    config = Config(name)

    # set the config as the current config
    config.set_current_config()
    _print_msg(f"Set config: {name}")


@cli.command(aliases=["uc", "update"])
@click.option("--name", type=str, default=None, required=True, help="Name of the config to update.")
@click.option(
    "--settings", 
    type=str, 
    required=True, 
    help="Parameters to update in format 'param1=value1,param2=value2'. Example: 'llmservice=openai,parallel=true,pdfprocessor=pdfplumber'"
)
def update_config(name: str, settings: str) -> None:
    """
    Update multiple parameters in an existing Palimpzest config.

    Parameters
    ----------
    name: str
        Name of the config to update
    params: str
        Comma-separated list of parameter=value pairs to update
    """
    from palimpzest.config import Config
    from palimpzest.constants import PZ_DIR

    # check that config exists
    if not os.path.exists(os.path.join(PZ_DIR, f"config_{name}.yaml")):
        raise InvalidCommandError(f"Config with name {name} does not exist.")

    # load the specified config
    config = Config(name)
    
    # Parse the params string into a dictionary
    try:
        param_pairs = settings.split(',')
        updates = {}
        for pair in param_pairs:
            if pair.strip() == "":
                continue
            param, value = pair.split('=')
            updates[param.strip()] = value.strip()
    except Exception as e:
        raise InvalidCommandError("Invalid params format. Use: param1=value1,param2=value2") from e
    
    # Update each parameter
    for param, value in updates.items():
        config.set(param, value)
    
    _print_msg(f"Updated config {name} with: {updates}")

def main():
    """
    Entrypoint for Palimpzest CLI tool implemented using Click.
    """
    cli.add_command(help)
    cli.add_command(init)
    cli.add_command(ls_data)
    cli.add_command(register_data)
    cli.add_command(rm_data)
    cli.add_command(clear_cache)
    cli.add_command(print_config)
    cli.add_command(create_config)
    cli.add_command(rm_config)
    cli.add_command(set_config)
    cli.add_command(update_config)
    cli()
