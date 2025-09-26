class AppConstants {
  // API Configuration
  static const String apiBaseUrl = 'http://192.168.1.100:5000'; // Change to your server IP
  static const Duration apiTimeout = Duration(seconds: 30);
  static const Duration connectionTimeout = Duration(seconds: 10);
  
  // API Endpoints
  static const String healthEndpoint = '/health';
  static const String analyzeBananaEndpoint = '/analyze_banana';
  static const String modelsInfoEndpoint = '/models/info';
  static const String testDetectionEndpoint = '/test_detection';
  static const String testFeaturesEndpoint = '/test_features';
  
  // Image Settings
  static const int maxImageSize = 1024;
  static const int imageQuality = 85;
  static const List<String> supportedFormats = ['jpg', 'jpeg', 'png'];
  static const int maxFileSizeMB = 10;
  
  // Camera Settings
  static const double aspectRatio = 3.0 / 4.0;
  static const int targetWidth = 640;
  static const int targetHeight = 480;
  
  // Storage Keys
  static const String keyAnalysisHistory = 'analysis_history';
  static const String keyUserPreferences = 'user_preferences';
  static const String keyFirstLaunch = 'first_launch';
  static const String keyServerUrl = 'server_url';
  
  // Ripeness Categories
  static const Map<String, RipenessInfo> ripenessCategories = {
    'green': RipenessInfo(
      name: 'Xanh',
      color: Color(0xFF4CAF50),
      description: 'Chuối còn xanh, chưa chín',
      daysRange: '5-7 ngày',
    ),
    'yellow': RipenessInfo(
      name: 'Vàng',
      color: Color(0xFFFFEB3B),
      description: 'Chuối chín, màu vàng đẹp',
      daysRange: '2-4 ngày',
    ),
    'spotted': RipenessInfo(
      name: 'Có đốm',
      color: Color(0xFFFF9800),
      description: 'Chuối rất chín, có đốm nâu',
      daysRange: '1-2 ngày',
    ),
    'brown': RipenessInfo(
      name: 'Nâu',
      color: Color(0xFF795548),
      description: 'Chuối quá chín, màu nâu',
      daysRange: '0-1 ngày',
    ),
  };
  
  // Animation Durations
  static const Duration shortAnimation = Duration(milliseconds: 300);
  static const Duration mediumAnimation = Duration(milliseconds: 500);
  static const Duration longAnimation = Duration(milliseconds: 800);
  
  // UI Constants
  static const double borderRadius = 12.0;
  static const double cardElevation = 4.0;
  static const double padding = 16.0;
  static const double smallPadding = 8.0;
  static const double largePadding = 24.0;
  
  // Error Messages
  static const String errorNoInternet = 'Không có kết nối internet';
  static const String errorServerUnreachable = 'Không thể kết nối đến server';
  static const String errorImageTooLarge = 'Ảnh quá lớn, vui lòng chọn ảnh khác';
  static const String errorInvalidImage = 'Định dạng ảnh không được hỗ trợ';
  static const String errorCameraPermission = 'Cần quyền truy cập camera';
  static const String errorStoragePermission = 'Cần quyền truy cập bộ nhớ';
  static const String errorAnalysisFailed = 'Phân tích thất bại, vui lòng thử lại';
  static const String errorNoBananaDetected = 'Không phát hiện chuối trong ảnh';
  
  // Success Messages
  static const String successAnalysisComplete = 'Phân tích hoàn tất!';
  static const String successImageSaved = 'Đã lưu ảnh thành công';
  static const String successHistoryCleared = 'Đã xóa lịch sử phân tích';
  
  // Tutorial Steps
  static const List<TutorialStep> tutorialSteps = [
    TutorialStep(
      title: 'Chụp ảnh chuối',
      description: 'Đặt chuối trong khung hình và nhấn nút chụp',
      icon: 'camera_alt',
    ),
    TutorialStep(
      title: 'AI phân tích',
      description: 'Hệ thống AI sẽ phân tích độ chín của chuối',
      icon: 'psychology',
    ),
    TutorialStep(
      title: 'Nhận kết quả',
      description: 'Xem thời gian sử dụng còn lại và lời khuyên',
      icon: 'schedule',
    ),
  ];
  
  // App Info
  static const String appName = 'Banana AI';
  static const String appVersion = '1.0.0';
  static const String appDescription = 'Ứng dụng phân tích độ chín chuối bằng AI';
  static const String supportEmail = 'support@bananaai.com';
  static const String privacyPolicyUrl = 'https://bananaai.com/privacy';
  static const String termsOfServiceUrl = 'https://bananaai.com/terms';
}

class RipenessInfo {
  final String name;
  final Color color;
  final String description;
  final String daysRange;
  
  const RipenessInfo({
    required this.name,
    required this.color,
    required this.description,
    required this.daysRange,
  });
}

class TutorialStep {
  final String title;
  final String description;
  final String icon;
  
  const TutorialStep({
    required this.title,
    required this.description,
    required this.icon,
  });
}

// Network Status
enum NetworkStatus { connected, disconnected, unknown }

// Analysis Status
enum AnalysisStatus { idle, loading, success, error }

// Image Source
enum ImageSourceType { camera, gallery }

// App Pages
enum AppPage { home, camera, analysis, history, settings }