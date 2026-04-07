package com.gemma4.llm;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Gemma4LLM";
    private static final int LLAMA_PORT = 8080;

    // Model files bundled in assets/
    // The main model is split into ~1 GB chunks (Android build tools can't handle >2 GB assets)
    private static final String MODEL_FILE = "gemma-4-E2B-it-Q4_K_M.gguf";
    private static final String[] MODEL_PARTS = {
        "gemma-4-E2B-it-Q4_K_M.gguf.part_aa",
        "gemma-4-E2B-it-Q4_K_M.gguf.part_ab",
        "gemma-4-E2B-it-Q4_K_M.gguf.part_ac",
    };
    private static final String MMPROJ_FILE = "mmproj-F16.gguf";

    private WebView webView;
    private LlamaService llamaService;
    private boolean serviceBound = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Edge-to-edge UI
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        getWindow().setStatusBarColor(0x00000000);
        getWindow().setNavigationBarColor(0x00000000);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        WindowInsetsControllerCompat controller =
                new WindowInsetsControllerCompat(getWindow(), getWindow().getDecorView());
        controller.setAppearanceLightStatusBars(false);
        controller.setAppearanceLightNavigationBars(false);

        // WebView as main UI
        webView = new WebView(this);
        setContentView(webView);
        setupWebView();

        // Inset handling for keyboard/nav bar
        View root = findViewById(android.R.id.content);
        root.setBackgroundColor(0xFF09090B);
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, windowInsets) -> {
            Insets bars = windowInsets.getInsets(
                    WindowInsetsCompat.Type.systemBars()
                            | WindowInsetsCompat.Type.displayCutout());
            Insets ime = windowInsets.getInsets(WindowInsetsCompat.Type.ime());
            int bottom = Math.max(bars.bottom, ime.bottom);
            v.setPadding(bars.left, bars.top, bars.right, bottom);
            return WindowInsetsCompat.CONSUMED;
        });

        // Notification permission (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
            }
        }

        // Extract model files (reassemble split chunks), then start service
        new Thread(() -> {
            reassembleModelIfNeeded();
            extractAssetIfNeeded(MMPROJ_FILE);
            runOnUiThread(this::startLlamaService);
        }).start();
    }

    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);

        webView.setBackgroundColor(0xFF09090B);
        webView.setClipToPadding(false);
        webView.setWebChromeClient(new WebChromeClient());
        webView.setWebViewClient(new WebViewClient());

        webView.loadUrl("file:///android_asset/index.html?apk=1&port=" + LLAMA_PORT);
    }

    private void reassembleModelIfNeeded() {
        File modelsDir = new File(getFilesDir(), "models");
        File modelFile = new File(modelsDir, MODEL_FILE);

        if (modelFile.exists()) {
            Log.i(TAG, "Model already assembled: " + MODEL_FILE + " (" + modelFile.length() + " bytes)");
            return;
        }

        Log.i(TAG, "Reassembling " + MODEL_PARTS.length + " chunks into " + MODEL_FILE);
        modelsDir.mkdirs();

        try (FileOutputStream fos = new FileOutputStream(modelFile)) {
            byte[] buffer = new byte[1024 * 1024]; // 1 MB
            long total = 0;
            for (String part : MODEL_PARTS) {
                Log.i(TAG, "  Reading chunk: " + part);
                try (InputStream is = getAssets().open(part)) {
                    int len;
                    while ((len = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, len);
                        total += len;
                    }
                }
            }
            Log.i(TAG, "Model assembled: " + (total / 1024 / 1024) + " MB");
        } catch (Exception e) {
            Log.e(TAG, "Failed to reassemble model", e);
            modelFile.delete();
        }
    }

    private void extractAssetIfNeeded(String assetName) {
        File modelsDir = new File(getFilesDir(), "models");
        File targetFile = new File(modelsDir, assetName);

        if (targetFile.exists()) {
            Log.i(TAG, "Already extracted: " + assetName + " (" + targetFile.length() + " bytes)");
            return;
        }

        Log.i(TAG, "Extracting " + assetName + " to " + targetFile.getAbsolutePath());
        modelsDir.mkdirs();

        try (InputStream is = getAssets().open(assetName);
             FileOutputStream fos = new FileOutputStream(targetFile)) {
            byte[] buffer = new byte[1024 * 1024]; // 1 MB chunks
            int len;
            long total = 0;
            while ((len = is.read(buffer)) != -1) {
                fos.write(buffer, 0, len);
                total += len;
            }
            Log.i(TAG, "Extracted " + assetName + ": " + (total / 1024 / 1024) + " MB");
        } catch (Exception e) {
            Log.e(TAG, "Failed to extract " + assetName, e);
            targetFile.delete();
        }
    }

    private void startLlamaService() {
        File modelsDir = new File(getFilesDir(), "models");
        Intent intent = new Intent(this, LlamaService.class);
        intent.putExtra("model_path", new File(modelsDir, MODEL_FILE).getAbsolutePath());
        intent.putExtra("mmproj_path", new File(modelsDir, MMPROJ_FILE).getAbsolutePath());
        intent.putExtra("port", LLAMA_PORT);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE);
    }

    private final ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            LlamaService.LocalBinder localBinder = (LlamaService.LocalBinder) binder;
            llamaService = localBinder.getService();
            serviceBound = true;
            Log.i(TAG, "Bound to LlamaService");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            serviceBound = false;
            llamaService = null;
        }
    };

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (serviceBound) {
            unbindService(serviceConnection);
            serviceBound = false;
        }
        stopService(new Intent(this, LlamaService.class));
    }
}
