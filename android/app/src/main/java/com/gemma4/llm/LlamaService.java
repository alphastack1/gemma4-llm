package com.gemma4.llm;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class LlamaService extends Service {
    private static final String TAG = "LlamaService";
    private static final String CHANNEL_ID = "gemma4_inference";
    private static final int NOTIFICATION_ID = 1;

    private final IBinder binder = new LocalBinder();
    private volatile Process llamaProcess;
    private volatile String modelPath;
    private volatile String mmprojPath;
    private int port = 8080;
    private volatile int threads = 0;     // 0 = auto-detect
    private volatile int ctxSize = 4096;
    private volatile boolean isReady = false;

    /**
     * Pick a thread count that maximizes tokens/sec on phones. Using ALL cores
     * hurts throughput on big.LITTLE (little cores stall the big ones at sync
     * points; memory bandwidth saturates past ~6). Heuristic: cores-2, clamped
     * to [4, 6] — approximates "big cores only" on modern 8-core phones.
     */
    private int autoThreadCount() {
        int cores = Runtime.getRuntime().availableProcessors();
        return Math.min(Math.max(4, cores - 2), 6);
    }

    public class LocalBinder extends Binder {
        LlamaService getService() {
            return LlamaService.this;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            modelPath = intent.getStringExtra("model_path");
            mmprojPath = intent.getStringExtra("mmproj_path");
            port = intent.getIntExtra("port", 8080);
            threads = intent.getIntExtra("threads", 0);
            ctxSize = intent.getIntExtra("ctx_size", 4096);
        }

        startForeground(NOTIFICATION_ID, buildNotification("Starting..."));
        new Thread(this::startLlamaServer).start();

        return START_NOT_STICKY;
    }

    private void startLlamaServer() {
        if (modelPath == null || !new File(modelPath).exists()) {
            Log.e(TAG, "Model not found: " + modelPath);
            updateNotification("Error: Model not found");
            return;
        }

        String nativeLibDir = getApplicationInfo().nativeLibraryDir;
        String serverBinary = nativeLibDir + "/libllama_server.so";

        if (!new File(serverBinary).exists()) {
            Log.e(TAG, "llama-server binary not found: " + serverBinary);
            updateNotification("Error: Binary not found");
            return;
        }

        int effectiveThreads = threads > 0 ? threads : autoThreadCount();
        Log.i(TAG, "Threads: " + effectiveThreads + " (cores="
                + Runtime.getRuntime().availableProcessors() + ", override=" + threads + ")");

        try {
            ProcessBuilder pb = new ProcessBuilder(
                serverBinary,
                "-m", modelPath,
                "--host", "127.0.0.1",
                "--port", String.valueOf(port),
                "-c", String.valueOf(ctxSize),
                "-t", String.valueOf(effectiveThreads),
                "--no-webui",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0"
            );

            // Add mmproj for vision if available
            if (mmprojPath != null && new File(mmprojPath).exists()) {
                pb.command().add("--mmproj");
                pb.command().add(mmprojPath);
                Log.i(TAG, "Vision enabled (mmproj loaded)");
            }

            pb.redirectErrorStream(true);
            pb.environment().put("LD_LIBRARY_PATH", nativeLibDir);

            Log.i(TAG, "Starting llama-server: " + String.join(" ", pb.command()));
            llamaProcess = pb.start();

            // Log output in background
            new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(llamaProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        Log.d(TAG, "llama: " + line);
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Output reader ended", e);
                }
            }).start();

            waitForReady();

        } catch (Exception e) {
            Log.e(TAG, "Failed to start llama-server", e);
            updateNotification("Error: " + e.getMessage());
        }
    }

    private void waitForReady() {
        updateNotification("Loading model...");

        for (int i = 0; i < 120; i++) {
            try {
                Thread.sleep(1000);

                // Check if process died
                if (llamaProcess != null) {
                    try {
                        int exit = llamaProcess.exitValue();
                        Log.e(TAG, "llama-server exited with code " + exit);
                        updateNotification("Error: Server crashed");
                        return;
                    } catch (IllegalThreadStateException e) {
                        // Still running — good
                    }
                }

                // Health check
                HttpURLConnection conn = (HttpURLConnection)
                        new URL("http://127.0.0.1:" + port + "/health").openConnection();
                conn.setConnectTimeout(1000);
                conn.setReadTimeout(1000);
                try {
                    if (conn.getResponseCode() == 200) {
                        isReady = true;
                        updateNotification("Ready — Gemma 4 E2B");
                        Log.i(TAG, "llama-server is ready on port " + port);
                        return;
                    }
                } finally {
                    conn.disconnect();
                }
            } catch (Exception e) {
                // Not ready yet
            }
        }

        Log.e(TAG, "llama-server did not become ready in 120s");
        updateNotification("Error: Timeout");
    }

    public boolean isReady() {
        return isReady;
    }

    public int getThreads()     { return threads > 0 ? threads : autoThreadCount(); }
    public int getCtxSize()     { return ctxSize; }
    public int getAutoThreads() { return autoThreadCount(); }
    public int getCpuCores()    { return Runtime.getRuntime().availableProcessors(); }

    /** Restart llama-server with new engine params. Safe to call from any thread. */
    public void reloadWithParams(int newThreads, int newCtx) {
        if (newThreads >= 0) this.threads = newThreads;
        if (newCtx > 0) this.ctxSize = newCtx;
        isReady = false;
        updateNotification("Applying settings...");
        if (llamaProcess != null) {
            llamaProcess.destroy();
            try {
                llamaProcess.waitFor();
            } catch (InterruptedException e) {
                llamaProcess.destroyForcibly();
            }
            llamaProcess = null;
        }
        new Thread(this::startLlamaServer).start();
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.notification_channel),
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Shows when Gemma4 LLM inference is running");
            NotificationManager nm = getSystemService(NotificationManager.class);
            nm.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification(String text) {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pending = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle(getString(R.string.notification_title))
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .setContentIntent(pending)
                .setOngoing(true)
                .setSilent(true)
                .build();
    }

    private void updateNotification(String text) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.notify(NOTIFICATION_ID, buildNotification(text));
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (llamaProcess != null) {
            Log.i(TAG, "Stopping llama-server...");
            llamaProcess.destroy();
            try {
                llamaProcess.waitFor();
            } catch (InterruptedException e) {
                llamaProcess.destroyForcibly();
            }
            llamaProcess = null;
            isReady = false;
            Log.i(TAG, "llama-server stopped.");
        }
    }
}
