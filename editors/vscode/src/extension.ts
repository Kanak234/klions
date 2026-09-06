import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration("klions");
  if (!config.get<boolean>("server.enable", true)) {
    return;
  }
  const command = config.get<string>("server.path", "klions-lsp");

  const serverOptions: ServerOptions = {
    run: { command, transport: TransportKind.stdio },
    debug: { command, transport: TransportKind.stdio },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "klions" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.kl"),
    },
  };

  client = new LanguageClient(
    "klions",
    "KLIONS Language Server",
    serverOptions,
    clientOptions
  );

  client.start().catch((err) => {
    vscode.window.showWarningMessage(
      `KLIONS: could not start '${command}'. Syntax highlighting still works. ` +
        `Set klions.server.path if the binary is elsewhere. (${err})`
    );
  });

  context.subscriptions.push({
    dispose: () => {
      client?.stop();
    },
  });
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
