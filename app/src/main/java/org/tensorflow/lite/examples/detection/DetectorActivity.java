/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Build;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import org.tensorflow.lite.examples.detection.env.SelfExpiringHashMap;
import org.tensorflow.lite.examples.detection.env.SelfExpiringMap;
import android.speech.tts.TextToSpeech;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.squareup.okhttp.Call;
import com.squareup.okhttp.Callback;
import com.squareup.okhttp.HttpUrl;
import com.squareup.okhttp.MediaType;
import com.squareup.okhttp.OkHttpClient;
import com.squareup.okhttp.Request;
import com.squareup.okhttp.RequestBody;
import com.squareup.okhttp.Response;

import java.util.Locale;
/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.75f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;
  private EditText getUserInput;
  private Button checkForObject;
  private Vibrator vibrator;


  private Classifier detector;

  private long azureProcessingTime;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  private TextToSpeech text2speech;

  private SelfExpiringMap<String, String> labelCache = new SelfExpiringHashMap<>();
  private final static int SLEEP_MULTIPLIER = 750;
  private String input_text;
  private List<String> CSVTokens = new ArrayList<String>();

  private List<Classifier.Recognition> azureResults = new ArrayList<>();

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    checkForObject = findViewById(R.id.object_button);
    getUserInput = findViewById(R.id.get_text);
    vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);

    checkForObject.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        CSVTokens.clear();
        input_text = String.valueOf(getUserInput.getText());
        List<String> tempTokens = Arrays.asList(input_text.split(","));
        if (!tempTokens.isEmpty()){
          for (final String str: tempTokens){
            String clean_tok = str.trim().toLowerCase();
            if (clean_tok.length() > 0) {
              CSVTokens.add(clean_tok);
            }
          }
        }
          if (Build.VERSION.SDK_INT >= 26) {
              vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
          } else {
              vibrator.vibrate(200);
          }
      }
    });

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    text2speech=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int status) {
        if(status != TextToSpeech.ERROR) {
          text2speech.setLanguage(Locale.US);
        }
      }
    });
  }

    private boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager
                = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        return activeNetworkInfo != null && activeNetworkInfo.isConnected();
    }

  public void postRequest(Bitmap cropCopyBitmap) {

    MediaType MEDIA_TYPE = MediaType.parse("application/octet-stream");
    String url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.1/analyze?visualFeatures=Objects";

    OkHttpClient client = new OkHttpClient();

    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    cropCopyBitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
    byte[] byteArray = stream.toByteArray();

    RequestBody body = RequestBody.create(MEDIA_TYPE, byteArray);

    Request request = new Request.Builder()
            .url(url)
            .post(body)
            .header("Accept", "application/json")
            .header("Ocp-Apim-Subscription-Key","<your_key_here>")
            .header("Content-Type", "application/octet-stream")
            .build();

    client.newCall(request).enqueue(new Callback() {
      @Override
      public void onFailure(Request request, IOException e) {
        String mMessage = e.getMessage().toString();
        LOGGER.w("failure Response", mMessage);
      }

      @Override
      public void onResponse(Response response) throws IOException {
        String mMessage = response.body().string();
        JsonObject fromJson = new Gson().fromJson(mMessage, JsonObject.class);
        JsonArray elements = fromJson.get("objects").getAsJsonArray();
        if (elements != null && elements.size() > 0) {
          for (int i = 0; i < elements.size(); i++) {
              JsonObject object = elements.get(i).getAsJsonObject();
              JsonObject rectangle = object.get("rectangle").getAsJsonObject();
              float x = Float.parseFloat(rectangle.get("x").toString());
              float y = Float.parseFloat(rectangle.get("y").toString());
              float w = Float.parseFloat(rectangle.get("w").toString()) + x;
              float h = Float.parseFloat(rectangle.get("h").toString()) + y;
              String name = object.get("object").toString();
              LOGGER.i("From cognitive service:: detected "+name);
              String confidence = object.get("confidence").toString();
              azureResults.add(new Classifier.Recognition(String.valueOf(i), name, Float.valueOf(confidence), new RectF(x, y, w, h)));
            }
        }
        LOGGER.i(mMessage);

      }

    });
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
            previewWidth,
            previewHeight,
            getLuminanceStride(),
            sensorOrientation,
            originalLuminance,
            timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
//    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }


    runInBackground(
            new Runnable() {
              @Override
              public void run() {


//                LOGGER.i("Running detection on image " + currTimestamp);
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas = new Canvas(cropCopyBitmap);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(2.0f);

                if(isNetworkAvailable()) {
                  if (SystemClock.uptimeMillis() / 1000 - azureProcessingTime > 60 / 20) {
                    postRequest(cropCopyBitmap);
                    azureProcessingTime = SystemClock.uptimeMillis() / 1000;
                  }
                }
                else
                    LOGGER.i("Network not connected");

                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                switch (MODE) {
                  case TF_OD_API:
                    minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                    break;
                }

                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();

                if (azureResults.size() != 0) {
                  results.addAll(azureResults);
                  azureResults.clear();
                }
                // speaking out the result in intelligent manner
                for (final Classifier.Recognition result : results) {
                  String label = result.getTitle();
                  final RectF location = result.getLocation();
                  //LOGGER.d("toSpeak " + label);
                  if (location != null && result.getConfidence() >= minimumConfidence &&
                          !labelCache.containsKey(label)) {
                    if (CSVTokens.isEmpty() || (!CSVTokens.isEmpty() && CSVTokens.contains(result.getTitle()))) {

                      canvas.drawRect(location, paint);
                      String direction = getGrid(location, new Size(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE));
                      cropToFrameTransform.mapRect(location);
//                      mappedRecognitions.add(result);
                      //int gridNum = getGridNum(location, DESIRED_PREVIEW_SIZE);
                      //LOGGER.d("gridNum " + gridNum);
                      //String toSpeak = label +" in "+gridNum;
                      //LOGGER.d("toSpeak " + toSpeak);

                      text2speech.speak(label+ direction, TextToSpeech.QUEUE_ADD, null);
                      Random rn = new Random();
                      int rand_int = rn.nextInt(6 - 2) + 2;
                      labelCache.put(label, label, rand_int * SLEEP_MULTIPLIER);
                    }
                  }

                }
                tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                trackingOverlay.postInvalidate();

                computingDetection = false;
              }
            });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private int getGridNum(RectF location, Size grid){
    int gridNum=7;
    Size cent_loc = new Size(  (int)(location.top + location.bottom)/2, (int)(location.left + location.right)/2);
    for (int i =1;i<=3;i++){
      for (int j=1;j<=3;j++){
        if(cent_loc.getWidth() <= i*grid.getWidth()/3 &&
           cent_loc.getHeight() <= j*grid.getHeight()/3 &&
           cent_loc.getHeight() > (j-1)*grid.getHeight()/3 &&
           cent_loc.getWidth() > (i-1)*grid.getHeight()/3)
          return (j-1)*3+j;
      }
    }
    LOGGER.i(String.valueOf(gridNum));
    return gridNum;
  }

  private String getGrid(RectF location, Size grid) {
    int gridNum = getGridNum(location, grid);
    switch (gridNum) {
      case 1:
      case 4:
      case 7:
        return "Left";
      case 2:
      case 5:
      case 8:
        return "Center";
      default:
        return "Right";
    }
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
