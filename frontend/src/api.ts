/**
 * API client for communicating with the FastAPI backend
 */

import axios from 'axios';
import type {
  UploadResponse,
  Layer1Response,
  Layer2Response,
  Layer3Response,
  Layer4Response,
  SessionStatus,
  LinkageResult,
} from './types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  /**
   * Upload PDF and STEP files to create a new session
   */
  async uploadFiles(
    pdfFile: File | null,
    stepFile: File | null,
    unit: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    if (pdfFile) formData.append('pdf_file', pdfFile);
    if (stepFile) formData.append('step_file', stepFile);
    formData.append('unit', unit);

    const response = await api.post<UploadResponse>('/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Run Layer 1: Azure DI layout extraction
   */
  async runLayer1(sessionId: string): Promise<Layer1Response> {
    const formData = new FormData();
    formData.append('session_id', sessionId);

    const response = await api.post<Layer1Response>('/api/pipeline/layer1', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Run Layer 2: GPT-4o vision enrichment
   */
  async runLayer2(sessionId: string): Promise<Layer2Response> {
    const formData = new FormData();
    formData.append('session_id', sessionId);

    const response = await api.post<Layer2Response>('/api/pipeline/layer2', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Run Layer 3: STEP file parsing (usually already done on upload)
   */
  async runLayer3(sessionId: string): Promise<Layer3Response> {
    const formData = new FormData();
    formData.append('session_id', sessionId);

    const response = await api.post<Layer3Response>('/api/pipeline/layer3', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Run Layer 4: Correlation between annotations and 3D features
   */
  async runLayer4(sessionId: string): Promise<Layer4Response> {
    const formData = new FormData();
    formData.append('session_id', sessionId);

    const response = await api.post<Layer4Response>('/api/pipeline/layer4', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Update user-edited mappings
   */
  async updateMappings(
    sessionId: string,
    mappings: Array<{ annotation_id: string; hole_ids: string[] }>
  ): Promise<{ status: string; updated_count: number }> {
    const response = await api.put(`/api/mappings/update`, 
      { mappings },
      { params: { session_id: sessionId } }
    );
    return response.data;
  },

  /**
   * Get drawing page image URL
   */
  getDrawingPageUrl(sessionId: string, pageNum: number): string {
    return `${API_BASE_URL}/api/drawing/page/${pageNum}?session_id=${sessionId}`;
  },

  /**
   * Get export JSON URL
   */
  getExportUrl(sessionId: string): string {
    return `${API_BASE_URL}/api/export?session_id=${sessionId}`;
  },

  /**
   * Get session status
   */
  async getSessionStatus(sessionId: string): Promise<SessionStatus> {
    const response = await api.get<SessionStatus>(`/api/session/${sessionId}`);
    return response.data;
  },

  /**
   * Delete session
   */
  async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/api/session/${sessionId}`);
  },

  /**
   * Export linkage result as JSON
   */
  async exportJson(sessionId: string): Promise<LinkageResult> {
    const response = await api.get<LinkageResult>(`/api/export`, {
      params: { session_id: sessionId },
    });
    return response.data;
  },
};
