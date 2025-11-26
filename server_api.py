#!/usr/bin/env python3
"""
서버 API: 라즈베리 파이에서 수집한 데이터를 받아 학습 수행
Flask 기반 REST API 서버
"""

import os
import pickle
import argparse
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

# 학습 관련 임포트
from train_with_teacher_forcing import TeacherForcingTrainer
from train_ppo import train_ppo
from train_imitation_rl import ImitationRLTrainer
from ppo_agent import PPOAgent

app = Flask(__name__)
CORS(app)  # CORS 허용 (라즈베리 파이에서 접근 가능하도록)

# 전역 변수
UPLOAD_FOLDER = 'uploaded_data'
MODEL_FOLDER = 'trained_models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """
    라즈베리 파이에서 수집한 데이터 업로드
    
    요청:
    - Content-Type: multipart/form-data
    - file: pickle 파일 (human_demos.pkl)
    
    응답:
    - status: success/error
    - file_path: 저장된 파일 경로
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"demos_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 데이터 검증
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            num_episodes = len(data.get('demonstrations', []))
            total_steps = sum(len(ep.get('states', [])) for ep in data.get('demonstrations', []))
        except Exception as e:
            return jsonify({'error': f'Invalid pickle file: {str(e)}'}), 400
        
        return jsonify({
            'status': 'success',
            'file_path': filepath,
            'filename': filename,
            'num_episodes': num_episodes,
            'total_steps': total_steps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/supervised', methods=['POST'])
def train_supervised():
    """
    Supervised Learning (Teacher Forcing) 학습 시작
    
    요청:
    - file_path: 업로드된 데이터 파일 경로
    - epochs: 학습 에폭 수 (기본: 100)
    - batch_size: 배치 크기 (기본: 64)
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    """
    try:
        data = request.json
        file_path = data.get('file_path')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 64)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file_path'}), 400
        
        # 데이터 로드
        with open(file_path, 'rb') as f:
            demo_data = pickle.load(f)
        
        demonstrations = demo_data.get('demonstrations', [])
        if len(demonstrations) == 0:
            return jsonify({'error': 'No demonstrations found'}), 400
        
        # 에이전트 생성
        agent = PPOAgent(
            state_dim=256,
            action_dim=5,
            discrete_action=True,
            use_recurrent=False
        )
        
        # Trainer 생성 및 학습
        trainer = TeacherForcingTrainer(agent, demonstrations)
        model_path = os.path.join(MODEL_FOLDER, f"pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
        trainer.pretrain(
            epochs=epochs,
            batch_size=batch_size,
            save_path=model_path
        )
        
        return jsonify({
            'status': 'success',
            'model_path': model_path,
            'epochs': epochs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/imitation_rl', methods=['POST'])
def train_imitation_rl_api():
    """
    Imitation Learning via Reinforcement Learning 학습 시작
    
    요청:
    - file_path: 업로드된 데모 데이터 파일 경로
    - model_path: 사전 학습된 모델 경로 (선택)
    - epochs: 학습 에폭 수 (기본: 100)
    - batch_size: 배치 크기 (기본: 64)
    - learning_rate: 학습률 (기본: 3e-4)
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    - final_match_rate: 최종 일치율
    """
    try:
        data = request.json
        file_path = data.get('file_path')
        model_path = data.get('model_path')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 64)
        learning_rate = data.get('learning_rate', 3e-4)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file_path'}), 400
        
        # Trainer 생성 및 학습
        trainer = ImitationRLTrainer(
            demos_path=file_path,
            model_path=model_path,
            device='cpu',  # 서버에서도 CPU 사용 (GPU가 있다면 'cuda'로 변경)
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        model_filename = f"imitation_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        
        # 학습 실행
        trainer.train(
            epochs=epochs,
            save_path=model_path,
            verbose=False  # 서버에서는 상세 출력 비활성화
        )
        
        # 최종 평가
        final_match_rate = trainer.evaluate()
        
        return jsonify({
            'status': 'success',
            'model_path': model_path,
            'final_match_rate': float(final_match_rate),
            'epochs': epochs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/ppo', methods=['POST'])
def train_ppo_api():
    """
    PPO 강화학습 시작
    
    요청:
    - model_path: 사전 학습된 모델 경로 (선택)
    - env_type: 환경 타입 (carracing/sim)
    - total_steps: 총 학습 스텝 수
    - ...
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    """
    try:
        data = request.json
        # TODO: PPO 학습 로직 구현
        return jsonify({
            'status': 'success',
            'message': 'PPO training started (async)'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/latest', methods=['GET'])
def get_latest_model():
    """
    최신 모델 다운로드
    
    응답:
    - 모델 파일 (.pth)
    """
    try:
        # MODEL_FOLDER에서 최신 모델 찾기
        model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
        if not model_files:
            return jsonify({'error': 'No model found'}), 404
        
        # 최신 파일 선택 (이름 기준)
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(MODEL_FOLDER, latest_model)
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=latest_model
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/list', methods=['GET'])
def list_models():
    """
    사용 가능한 모델 목록 조회
    
    응답:
    - models: 모델 파일 목록
    """
    try:
        model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
        model_info = []
        
        for model_file in sorted(model_files, reverse=True):
            model_path = os.path.join(MODEL_FOLDER, model_file)
            stat = os.stat(model_path)
            model_info.append({
                'filename': model_file,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return jsonify({
            'models': model_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inference', methods=['POST'])
def inference():
    """
    실시간 추론 (선택 사항)
    
    요청:
    - state: 256차원 상태 벡터 (16x16 이미지)
    - model_path: 사용할 모델 경로 (선택, 기본: 최신 모델)
    
    응답:
    - action: 추론된 액션 (0-4)
    - log_prob: 로그 확률
    - value: 상태 가치
    """
    try:
        data = request.json
        state = data.get('state')
        model_path = data.get('model_path')
        
        if state is None:
            return jsonify({'error': 'No state provided'}), 400
        
        # 모델 로드
        if model_path is None:
            # 최신 모델 사용
            model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
            if not model_files:
                return jsonify({'error': 'No model found'}), 404
            model_path = os.path.join(MODEL_FOLDER, sorted(model_files)[-1])
        
        # 에이전트 로드 및 추론
        agent = PPOAgent(
            state_dim=256,
            action_dim=5,
            discrete_action=True
        )
        agent.load(model_path)
        
        # 추론
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = agent.actor_critic.get_action(state_tensor)
        
        return jsonify({
            'action': int(action.item()),
            'log_prob': float(log_prob.item()),
            'value': float(value.item())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description='RC Car 학습 서버 API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='서버 호스트 (기본: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='서버 포트 (기본: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')
    
    args = parser.parse_args()
    
    print(f"🚀 서버 시작: http://{args.host}:{args.port}")
    print(f"📁 업로드 폴더: {UPLOAD_FOLDER}")
    print(f"📁 모델 폴더: {MODEL_FOLDER}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

