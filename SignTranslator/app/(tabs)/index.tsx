import { View, Text, Button, StyleSheet, ActivityIndicator, Image, ScrollView, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video, ResizeMode } from 'expo-av';
import { useState } from 'react';

// REPLACE WITH YOUR IP
const API_URL = "http://10.215.64.229:8000/predict-video"; 

export default function HomeScreen() {
  const [video, setVideo] = useState<string | null>(null);
  const [result, setResult] = useState<string>('');
  const [confidence, setConfidence] = useState<string>('');
  const [replayGif, setReplayGif] = useState<string | null>(null); // For the GIF
  const [loading, setLoading] = useState(false);

  const openCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission", "Camera access is required.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: true,
      quality: 0.5,
      videoMaxDuration: 4, 
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setVideo(uri);
      setResult(''); 
      setConfidence('');
      setReplayGif(null); // Clear old GIF
      uploadVideo(uri);
    }
  };

  const uploadVideo = async (uri: string) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: uri,
        name: 'upload.mp4',
        type: 'video/mp4',
      } as any);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = await response.json();
      
      if (data.error) {
        setResult("Error");
        setConfidence(data.error);
      } else {
        setResult(data.prediction);
        setConfidence(data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '');
        
        // Load the GIF
        if (data.replay_gif) {
          setReplayGif(`data:image/gif;base64,${data.replay_gif}`);
        }
      }
    } catch (error) {
      setResult("Network Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Sign Language AI</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Your Input</Text>
        <View style={styles.videoContainer}>
          {video ? (
            <Video
              source={{ uri: video }}
              style={styles.video}
              resizeMode={ResizeMode.COVER}
              shouldPlay
              isLooping
            />
          ) : (
            <View style={styles.placeholder}><Text style={{color:'#999'}}>No Video</Text></View>
          )}
        </View>
        <Button title={loading ? "Analyzing..." : "Record Sign"} onPress={openCamera} disabled={loading} />
      </View>

      {loading && <ActivityIndicator size="large" color="#007bff" style={{marginTop: 20}} />}

      {result !== '' && !loading && (
        <View style={[styles.card, styles.resultCard]}>
          <Text style={styles.resultLabel}>Prediction:</Text>
          <Text style={styles.resultText}>{result}</Text>
          {confidence !== '' && <Text style={styles.confText}>Confidence: {confidence}</Text>}

          {/* AI REPLAY SECTION */}
          {replayGif && (
            <View style={styles.debugContainer}>
              <Text style={styles.debugLabel}>AI Processed Replay:</Text>
              <Image 
                source={{ uri: replayGif }} 
                style={styles.debugImage} 
                resizeMode="contain"
              />
              <Text style={{fontSize: 10, color: '#999', marginTop: 5}}>
                (Landmarks tracked by MediaPipe)
              </Text>
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, alignItems: 'center', padding: 20, backgroundColor: '#f4f6f8' },
  title: { fontSize: 26, fontWeight: 'bold', marginBottom: 20, color: '#2c3e50' },
  card: { width: '100%', backgroundColor: 'white', borderRadius: 15, padding: 15, marginBottom: 20, elevation: 4, alignItems: 'center' },
  resultCard: { borderLeftWidth: 5, borderLeftColor: '#27ae60' },
  cardTitle: { fontSize: 16, fontWeight: '600', marginBottom: 10, color: '#7f8c8d' },
  videoContainer: { width: 220, height: 220, borderRadius: 12, overflow: 'hidden', backgroundColor: '#ecf0f1', marginBottom: 15 },
  video: { width: '100%', height: '100%' },
  placeholder: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  resultLabel: { fontSize: 14, textTransform: 'uppercase', color: '#7f8c8d', fontWeight: 'bold' },
  resultText: { fontSize: 32, fontWeight: 'bold', color: '#27ae60', marginVertical: 5 },
  confText: { fontSize: 14, color: '#95a5a6', marginBottom: 15 },
  debugContainer: { marginTop: 15, width: '100%', alignItems: 'center', borderTopWidth: 1, borderTopColor: '#eee', paddingTop: 15 },
  debugLabel: { fontSize: 12, color: '#34495e', marginBottom: 8, fontStyle: 'italic', fontWeight: 'bold' },
  debugImage: { width: 250, height: 200, borderRadius: 8, backgroundColor: '#000' }
});