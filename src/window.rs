#![allow(missing_docs)]

use std::ffi::{c_int, c_void};
use std::mem::{ManuallyDrop};
use crate::{ffi, Rank};
use crate::topology::UserGroup;
use crate::traits::{AsRaw, Equivalence};

pub struct CreatedWindow<'a, T> where T: Equivalence {
    pub window_vec: &'a mut Vec<T>,
    pub window_handle: ffi::MPI_Win
}

pub struct AllocatedWindow<T> where T: Equivalence {
    pub window_vec: ManuallyDrop<Vec<T>>,
    pub window_handle: ffi::MPI_Win
}

pub trait Communication<T> {
    fn put_from_vector(&self, origin: &Vec<T>, target_rank: usize);
    fn get_from_vector(&self, origin: &mut Vec<T>, target_rank: usize);
    fn put(&self, origin: &Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize);
    fn get(&self, origin: &mut Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize);
    fn put_whole_vector(&self, target_rank: usize);
    fn get_whole_vector(&mut self, target_rank: usize);
}

pub trait Synchronization {
    fn fence(&self);
    fn post(&self, group: &UserGroup);
    fn start(&self, group: &UserGroup);
    fn complete(&self);
    fn wait(&self);
    fn exclusive_lock(&self, rank: Rank);
    fn unlock(&self, rank: Rank);
}

impl<T> Communication<T> for AllocatedWindow<T> where T: Equivalence {
    fn put_from_vector(&self, origin: &Vec<T>, target_rank: usize) {
        common_put(origin, origin.len(), target_rank, 0, origin.len(), self.window_handle);
    }

    fn get_from_vector(&self, origin: &mut Vec<T>, target_rank: usize) {
        common_get(origin, origin.len(), target_rank, 0, origin.len(), self.window_handle);
    }

    fn put(&self, origin: &Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        common_put(origin, origin_count, target_rank, target_disp, target_count, self.window_handle);
    }

    fn get(&self, origin: &mut Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        common_get(origin, origin_count, target_rank, target_disp, target_count, self.window_handle);
    }

    fn put_whole_vector(&self, target_rank: usize) {
        common_put(&self.window_vec, self.window_vec.len(), target_rank, 0, self.window_vec.len(), self.window_handle);
    }

    fn get_whole_vector(&mut self, target_rank: usize) {
        let len = self.window_vec.len();
        common_get(&mut self.window_vec, len, target_rank, 0, len, self.window_handle);
    }
}

impl <T> Synchronization for AllocatedWindow<T> where T: Equivalence{
    fn fence(&self) {
        common_fence(self.window_handle);
    }

    fn post(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_post(group.as_raw(), 0, self.window_handle);
        }
    }

    fn start(&self, group: &UserGroup) {
        unsafe {
            ffi::MPI_Win_start(group.as_raw(), 0, self.window_handle);
        }
    }

    fn complete(&self) {
        unsafe {
            ffi::MPI_Win_complete(self.window_handle);
        }
    }

    fn wait(&self) {
        unsafe {
            ffi::MPI_Win_wait(self.window_handle);
        }
    }

    fn exclusive_lock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_lock(ffi::MPI_LOCK_EXCLUSIVE as c_int, rank as c_int, 0, self.window_handle);
        }
    }

    fn unlock(&self, rank: Rank) {
        unsafe {
            ffi::MPI_Win_unlock(rank as c_int, self.window_handle);
        }
    }
}

fn common_put<T>(origin: &Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize, window: ffi::MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Put(
            origin.as_ptr() as *const c_void,
            origin_count as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            target_disp as ffi::MPI_Aint,
            target_count as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}

fn common_get<T>(origin: &mut Vec<T>, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize, window: ffi::MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Get(
            origin.as_mut_ptr() as *mut c_void,
            origin_count as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            target_disp as ffi::MPI_Aint,
            target_count as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}

fn common_fence(window: ffi::MPI_Win) {
    unsafe {
        ffi::MPI_Win_fence(0, window);
    }
}

impl<T> Drop for AllocatedWindow<T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_handle);
        }
    }
}

impl<'a, T> Drop for CreatedWindow<'a, T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_handle);
        }
    }
}